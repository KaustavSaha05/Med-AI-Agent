"""
Replace patient names in `data/synthetic_patients/patients.json` with Indian names,
keeping the first four entries unchanged.

Usage:
    python scripts/replace_patient_names.py

This script makes a backup `patients.json.bak` before editing.
"""
import json
from pathlib import Path
from faker import Faker
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
PATIENTS_FILE = BASE_DIR / "data" / "synthetic_patients" / "patients.json"

INDIAN_NAMES_POOL = [
    # Common Indian names sample (male & female, full names)
    "Amit Kumar", "Priya Singh", "Rohan Sharma", "Sneha Patel",
    "Vikas Gupta", "Anita Desai", "Rahul Mehta", "Pooja Nair",
    "Sanjay Reddy", "Neha Kapoor", "Manish Iyer", "Deepa Menon",
    "Arjun Rao", "Kavita Joshi", "Vivek Shah", "Madhuri Chatterjee",
    "Suresh Kumar", "Anjali Bose", "Karan Verma", "Ritu Ghosh",
    "Rakesh Sinha", "Ishita Banerjee", "Tanmay Kulkarni", "Meera Bhattacharya",
    "Devansh Bhatia", "Nidhi Rao", "Siddharth Mitra", "Priyanka Choudhury",
    "Aarav Singh", "Lakshmi Nair", "Kunal Dutta", "Sana Khan",
]


def load_patients(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_patients(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    if not PATIENTS_FILE.exists():
        print(f"patients.json not found at {PATIENTS_FILE}")
        return

    backup = PATIENTS_FILE.with_suffix('.json.bak')
    shutil.copy2(PATIENTS_FILE, backup)
    print(f"Backup created: {backup}")

    patients = load_patients(PATIENTS_FILE)

    fake = Faker('en_IN')

    # Keep first 4 names unchanged
    for i in range(len(patients)):
        if i < 4:
            continue
        # Prefer using curated pool; fallback to Faker
        name = None
        try:
            name = INDIAN_NAMES_POOL[(i - 4) % len(INDIAN_NAMES_POOL)]
        except Exception:
            name = fake.name()
        patients[i]['name'] = name

    save_patients(PATIENTS_FILE, patients)
    print(f"Updated {len(patients)-4} patient names (kept first 4 unchanged).")

    # Print first 8 names for verification
    print("First 8 patient names:")
    for p in patients[:8]:
        print(f"  {p['patient_id']}: {p['name']}")


if __name__ == '__main__':
    main()
