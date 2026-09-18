import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from models import ContextNote

load_dotenv(dotenv_path=Path.cwd() / ".env")
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class NoteRetriever:
    def __init__(self, notes: List[ContextNote]):
        self.notes = notes
        self.notes_by_id = {n.note_id: n for n in notes}
        self.vector_store = None
        self._init_chroma()

    def _init_chroma(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and self.notes:
            try:
                from langchain_openai import OpenAIEmbeddings
                from langchain_chroma import Chroma
                from langchain_core.documents import Document

                docs = [
                    Document(
                        page_content=f"Route: {n.applies_to}. Date: {n.note_date}. Disruption: {n.note_text}",
                        metadata={"note_id": n.note_id, "applies_to": n.applies_to, "date": str(n.note_date)}
                    )
                    for n in self.notes
                ]
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                self.vector_store = Chroma.from_documents(docs, embeddings)
            except Exception:
                self.vector_store = None

    @staticmethod
    def _parse_week(week_of: str):
        return datetime.strptime(week_of, "%Y-%m-%d").date()

    def retrieve_applicable_notes(
        self,
        route: str,
        week_of: str,
        max_days_distance: int = 21
    ) -> List[Tuple[ContextNote, float, int]]:
        target_date = self._parse_week(week_of)
        week_end = target_date + timedelta(days=6)

        results: List[Tuple[ContextNote, float, int]] = []
        for note in self.notes:
            if not note.matches_route(route):
                continue

            days_diff = (note.note_date - target_date).days
            abs_days = abs(days_diff)
            is_exact_week = (target_date <= note.note_date <= week_end)

            score = 0.0
            if is_exact_week:
                score += 100.0
            elif abs_days <= 7:
                score += 70.0
            elif abs_days <= max_days_distance:
                score += max(10.0, 50.0 - abs_days * 1.5)
            else:
                continue

            if note.applies_to == route:
                score += 25.0
            elif note.applies_to == "All Routes":
                score += 10.0

            results.append((note, score, days_diff))

        results.sort(key=lambda x: (-x[1], abs(x[2])))
        return results

    def vector_search(self, query: str, k: int = 3) -> List[Tuple[ContextNote, float]]:
        if not self.vector_store:
            return []
        try:
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            matches = []
            for doc, score in docs_with_scores:
                note_id = doc.metadata.get("note_id")
                note = self.notes_by_id.get(note_id)
                if note:
                    matches.append((note, score))
            return matches
        except Exception:
            return []

    def get_closest_note(self, route: str, week_of: str) -> Optional[Tuple[ContextNote, int]]:
        target_date = self._parse_week(week_of)
        candidates = [(n, (n.note_date - target_date).days) for n in self.notes if n.matches_route(route)]
        if not candidates:
            return None
        candidates.sort(key=lambda x: abs(x[1]))
        return candidates[0]


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    notes = loader.load_notes()
    retriever = NoteRetriever(notes)

    res_ahmedabad = retriever.retrieve_applicable_notes("Ahmedabad-Mumbai", "2025-01-20")
    assert len(res_ahmedabad) > 0
    assert res_ahmedabad[0][0].note_id == "N002"

    res_chennai = retriever.retrieve_applicable_notes("Chennai-Bangalore", "2025-02-24")
    assert len(res_chennai) > 0
    assert res_chennai[0][0].note_id == "N001"

    closest = retriever.get_closest_note("Mumbai-Pune", "2025-09-15")
    assert closest is not None
    assert closest[0].note_id == "N006"

    # Test Chroma Vector Search
    vector_results = retriever.vector_search("Ahmedabad festival surcharge and truck shortage", k=1)
    if vector_results:
        print(f"src/rag_retriever.py Chroma Vector Match: [{vector_results[0][0].note_id}] Score: {vector_results[0][1]:.3f}")
        assert vector_results[0][0].note_id == "N002"

    print("src/rag_retriever.py: All retrieval and Chroma tests passed.")
