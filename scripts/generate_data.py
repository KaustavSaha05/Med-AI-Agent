"""
Generate synthetic patient data with medication records.
This script creates realistic patient data for testing the medication management system.
"""
import argparse
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import SYNTHETIC_PATIENTS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "data_generation.log")
fake = Faker()

# Specific Indian names to include for the first four patients
INDIAN_NAMES = [
    "Aritra Chowdhury",
    "Kaustav Saha",
    "Soumyendu Sekhar Bandyopadhyay",
    "Susovan Jana",
]

# Common medications and conditions
MEDICATIONS = {
    "Hypertension": ["Lisinopril", "Amlodipine", "Losartan", "Metoprolol"],
    "Diabetes": ["Metformin", "Glipizide", "Insulin", "Januvia"],
    "High Cholesterol": ["Atorvastatin", "Simvastatin", "Rosuvastatin"],
    "Depression": ["Sertraline", "Escitalopram", "Fluoxetine", "Bupropion"],
    "Anxiety": ["Alprazolam", "Lorazepam", "Buspirone"],
    "Asthma": ["Albuterol", "Fluticasone", "Montelukast"],
    "GERD": ["Omeprazole", "Pantoprazole", "Ranitidine"],
    "Arthritis": ["Ibuprofen", "Naproxen", "Celecoxib", "Prednisone"]
}

DRUG_INTERACTIONS = {
    ("Lisinopril", "Ibuprofen"): "May reduce effectiveness of blood pressure medication",
    ("Metformin", "Alcohol"): "Increased risk of lactic acidosis",
    ("Sertraline", "Alprazolam"): "Increased sedation and respiratory depression",
    ("Atorvastatin", "Grapefruit"): "Increased risk of muscle damage",
    ("Warfarin", "Aspirin"): "Increased bleeding risk"
}

def generate_patient(patient_id: int, forced_name: str | None = None) -> dict:
    """Generate a single synthetic patient with medical history."""
    age = np.random.randint(25, 85)
    gender = np.random.choice(["Male", "Female"])
    
    # Assign 1-4 conditions based on age
    num_conditions = min(1 + int(age / 25), 4)
    conditions = np.random.choice(list(MEDICATIONS.keys()), num_conditions, replace=False)
    
    # Assign medications based on conditions
    medications = []
    for condition in conditions:
        med = np.random.choice(MEDICATIONS[condition])
        dosage = f"{np.random.choice([5, 10, 20, 25, 50, 100])}mg"
        frequency = np.random.choice(["once daily", "twice daily", "three times daily"])
        medications.append({
            "name": med,
            "dosage": dosage,
            "frequency": frequency,
            "condition": condition,
            "start_date": (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime("%Y-%m-%d")
        })
    
    # Generate adherence pattern
    baseline_adherence = np.random.uniform(0.3, 0.7)
    
    # Select allergies
    allergy_rand = np.random.random()
    if allergy_rand < 0.7:
        allergies = []
    elif allergy_rand < 0.8:
        allergies = ["Penicillin"]
    elif allergy_rand < 0.9:
        allergies = ["Sulfa drugs"]
    else:
        allergies = ["Aspirin"]
    
    patient = {
        "patient_id": f"P{patient_id:04d}",
        "name": forced_name or fake.name(),
        "age": age,
        "gender": gender,
        "email": fake.email(),
        "phone": fake.phone_number(),
        "conditions": list(conditions),
        "medications": medications,
        "baseline_adherence": round(baseline_adherence, 2),
        "allergies": allergies,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return patient

def generate_medication_schedule(patient: dict, days: int = 30) -> pd.DataFrame:
    """Generate a medication schedule for a patient."""
    records = []
    start_date = datetime.now() - timedelta(days=days)
    
    for med in patient["medications"]:
        # Determine doses per day
        if "once" in med["frequency"]:
            doses_per_day = 1
        elif "twice" in med["frequency"]:
            doses_per_day = 2
        elif "three" in med["frequency"]:
            doses_per_day = 3
        else:
            doses_per_day = 1
        
        # Generate schedule
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            for dose_num in range(doses_per_day):
                # Simulate adherence based on patient's baseline
                taken = np.random.random() < patient["baseline_adherence"]
                
                # Add some realistic patterns (e.g., better adherence in morning)
                if dose_num == 0:
                    taken = np.random.random() < (patient["baseline_adherence"] + 0.1)
                
                records.append({
                    "patient_id": patient["patient_id"],
                    "medication": med["name"],
                    "dosage": med["dosage"],
                    "scheduled_time": (current_date + timedelta(hours=8 + dose_num * 8)).strftime("%Y-%m-%d %H:%M:%S"),
                    "taken": taken,
                    "taken_time": (current_date + timedelta(hours=8 + dose_num * 8, minutes=np.random.randint(-30, 120))).strftime("%Y-%m-%d %H:%M:%S") if taken else None
                })
    
    return pd.DataFrame(records)

def main(num_patients: int = 100, schedule_days: int = 30):
    """Generate synthetic patient data."""
    logger.info(f"Generating data for {num_patients} patients with {schedule_days} days of medication schedules...")
    
    # Generate patients (first four use specified Indian names)
    patients = []
    for i in range(1, num_patients + 1):
        name = INDIAN_NAMES[i-1] if i-1 < len(INDIAN_NAMES) else None
        patients.append(generate_patient(i, forced_name=name))
    
    # Save patients data
    patients_file = SYNTHETIC_PATIENTS_DIR / "patients.json"
    with open(patients_file, 'w') as f:
        json.dump(patients, f, indent=2)
    logger.info(f"Saved {len(patients)} patients to {patients_file}")
    
    # Generate and save medication schedules
    all_schedules = []
    for patient in patients:
        schedule = generate_medication_schedule(patient, schedule_days)
        all_schedules.append(schedule)
    
    combined_schedule = pd.concat(all_schedules, ignore_index=True)
    schedule_file = SYNTHETIC_PATIENTS_DIR / "medication_schedules.csv"
    combined_schedule.to_csv(schedule_file, index=False)
    logger.info(f"Saved {len(combined_schedule)} medication records to {schedule_file}")
    
    # Generate summary statistics
    summary = {
        "total_patients": num_patients,
        "total_medication_records": len(combined_schedule),
        "average_adherence": round(combined_schedule["taken"].mean(), 3),
        "conditions_distribution": pd.DataFrame([p["conditions"] for p in patients]).stack().value_counts().to_dict(),
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = SYNTHETIC_PATIENTS_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_file}")
    
    logger.info("Data generation completed successfully!")
    logger.info(f"   - Average baseline adherence: {summary['average_adherence']:.1%}")
    logger.info(f"   - Total medication records: {summary['total_medication_records']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic patient medication data")
    parser.add_argument("--patients", type=int, default=100, help="Number of patients to generate")
    parser.add_argument("--days", type=int, default=30, help="Days of medication schedule history")
    args = parser.parse_args()
    
    main(num_patients=args.patients, schedule_days=args.days)
