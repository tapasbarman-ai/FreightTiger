import sys
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models import ShipmentRecord, ContextNote


class DataLoader:
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        elif (Path.cwd() / "data").exists():
            self.data_dir = Path.cwd() / "data"
        elif (Path(__file__).parent.parent / "data").exists():
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path.cwd()

    def load_shipments(self, filename: str = "shipment_records.csv") -> List[ShipmentRecord]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            filepath = Path.cwd() / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Shipment records file not found: {filepath}")

        df = pd.read_csv(filepath)
        records: List[ShipmentRecord] = []
        for row in df.itertuples(index=False):
            parsed_date = datetime.strptime(row.shipment_date, "%Y-%m-%d").date()
            record = ShipmentRecord(
                shipment_id=str(row.shipment_id),
                origin=str(row.origin),
                destination=str(row.destination),
                route_type=str(row.route_type),
                material=str(row.material),
                quantity_tonnes=float(row.quantity_tonnes),
                distance_km=float(row.distance_km),
                freight_cost_inr=float(row.freight_cost_inr),
                shipment_date=parsed_date,
                transporter=str(row.transporter)
            )
            records.append(record)
        return records

    def load_notes(self, filename: str = "context_notes.csv") -> List[ContextNote]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            filepath = Path.cwd() / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Context notes file not found: {filepath}")

        df = pd.read_csv(filepath)
        notes: List[ContextNote] = []
        for row in df.itertuples(index=False):
            parsed_date = datetime.strptime(row.date, "%Y-%m-%d").date()
            note = ContextNote(
                note_id=str(row.note_id).strip(),
                note_date=parsed_date,
                applies_to=str(row.applies_to).strip(),
                note_text=str(row.note).strip()
            )
            notes.append(note)
        return notes


if __name__ == "__main__":
    loader = DataLoader()
    shipments = loader.load_shipments()
    notes = loader.load_notes()

    print(f"src/data_loader.py: Loaded {len(shipments)} shipment records from {loader.data_dir}.")
    print(f"src/data_loader.py: Loaded {len(notes)} context notes from {loader.data_dir}.")
    assert len(shipments) > 0
    assert len(notes) > 0
