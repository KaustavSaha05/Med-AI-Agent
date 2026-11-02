"""
Full System Demonstration - Agentic AI Medication Management
Demonstrates all capabilities of the multi-agent system
"""
import sys
from pathlib import Path
import json
import os
import pandas as pd
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for emojis on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

sys.path.append(str(Path(__file__).parent))

from src.agents.medication_manager import MedicationManagerAgent
from src.agents.nurse_agent import NurseAgent
from src.agents.monitor_agent import MonitorAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def load_patient_data():
    """Load patient and medication data."""
    with open('data/synthetic_patients/patients.json', 'r') as f:
        patients = json.load(f)
    
    medication_df = pd.read_csv('data/synthetic_patients/medication_schedules.csv')
    
    return patients, medication_df

def demo_patient_analysis(patient_id, patients, medication_df):
    """Demonstrate full analysis for a single patient."""
    print_header(f"PATIENT ANALYSIS: {patient_id}")
    
    # Get patient data
    patient = next((p for p in patients if p['patient_id'] == patient_id), None)
    if not patient:
        print(f"❌ Patient {patient_id} not found")
        return
    
    patient_meds = medication_df[medication_df['patient_id'] == patient_id]
    
    # Display patient info
    print("👤 PATIENT PROFILE")
    print(f"   Name: {patient['name']}")
    print(f"   Age: {patient['age']} | Gender: {patient['gender']}")
    print(f"   Conditions: {', '.join(patient['conditions'])}")
    
    # Extract medication names from the medications list
    med_names = [med['name'] for med in patient['medications']]
    print(f"   Current Medications: {', '.join(med_names)}")
    
    allergies = patient['allergies'] if patient['allergies'] else 'None'
    print(f"   Allergies: {', '.join(allergies) if isinstance(allergies, list) else allergies}")
    
    # Calculate adherence
    adherence_rate = (patient_meds['taken'].sum() / len(patient_meds)) * 100
    print(f"\n📊 ADHERENCE RATE: {adherence_rate:.1f}%")
    
    return patient, patient_meds, adherence_rate

def main():
    """Run full system demonstration."""
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("="  + "  AGENTIC AI MEDICATION MANAGEMENT SYSTEM - FULL DEMO  ".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)
    
    print_header("SYSTEM INITIALIZATION")
    
    # Initialize agents
    print("🔧 Initializing agents...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    medication_manager = MedicationManagerAgent(google_api_key=google_api_key)
    nurse_agent = NurseAgent(google_api_key=google_api_key)
    monitor_agent = MonitorAgent()
    print("✅ All agents initialized successfully!\n")
    
    # Load data
    print("📊 Loading patient database...")
    patients, medication_df = load_patient_data()
    print(f"✅ Loaded {len(patients)} patients with {len(medication_df)} medication records\n")
    
    # Demonstrate on multiple patients
    demo_patients = ['P0001', 'P0002', 'P0003']
    
    for patient_id in demo_patients:
        patient, patient_meds, adherence_rate = demo_patient_analysis(patient_id, patients, medication_df)
        
        if patient is None:
            continue
        
        # 1. Drug Interaction Check
        print("\n💊 DRUG INTERACTION CHECK (Monitor Agent)")
        print("-" * 70)
        med_names = [med['name'] for med in patient['medications']]
        interactions = monitor_agent.check_medication_interactions(med_names)
        if interactions:
            print(f"⚠️  Found {len(interactions)} potential interactions:")
            for interaction in interactions:
                print(f"   • {interaction}")
        else:
            print("✅ No known drug interactions detected")
        
        # 2. Medication Knowledge Query
        print("\n🤖 MEDICATION INFORMATION (Medication Manager + RAG)")
        print("-" * 70)
        query = f"What are the key things I should know about {med_names[0]}?"
        print(f"Q: {query}\n")
        
        patient_context = {
            'patient_id': patient['patient_id'],
            'medications': med_names,
            'conditions': patient['conditions'],
            'allergies': patient['allergies']
        }
        
        result = medication_manager.process_query(query, patient_context)
        
        # Handle different response formats
        if isinstance(result, dict) and 'answer' in result:
            response_text = result['answer']
        elif isinstance(result, dict) and 'response' in result:
            response_text = result['response']
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result)
            
        print(f"A: {response_text[:500]}...\n")

        # 3. Adherence Coaching
        print("\n💬 ADHERENCE COACHING (Nurse Agent)")
        print("-" * 70)
        
        # Analyze adherence patterns
        scheduled_dt = pd.to_datetime(patient_meds['scheduled_time'])
        hours = scheduled_dt.dt.hour
        time_of_day_series = hours.apply(lambda h: 'morning' if h < 12 else ('afternoon' if h < 17 else 'evening'))
        patient_meds_copy = patient_meds.copy()
        patient_meds_copy['time_of_day'] = time_of_day_series

        recent_missed = patient_meds_copy[patient_meds_copy['taken'] == False].sort_values('scheduled_time').tail(7)
        missed_doses = [
            {
                'medication': row['medication'],
                'time_of_day': row['time_of_day'],
                'day_of_week': pd.to_datetime(row['scheduled_time']).strftime('%A')
            }
            for _, row in recent_missed.iterrows()
        ]
        
        coaching = nurse_agent.provide_adherence_coaching(
            patient_context=patient_context,
            adherence_rate=adherence_rate / 100.0,
            missed_doses=missed_doses
        )
        
        pattern = coaching.get('pattern', 'Consistent adherence pattern')
        coaching_msg = coaching.get('coaching_message', '')
        print(f"Current Adherence: {adherence_rate:.1f}%")
        print(f"Pattern: {pattern}")
        print(f"\nCoaching Message:\n{coaching_msg[:400]}...\n")
        
        # 4. Monitoring Report
        print("\n📈 MONITORING ANALYSIS (Monitor Agent)")
        print("-" * 70)
        
        adherence_history = [
            {
                'scheduled_time': str(row['scheduled_time']),
                'taken': bool(row['taken']),
                'medication': row['medication']
            }
            for _, row in patient_meds.iterrows()
        ]
        report = monitor_agent.generate_monitoring_report(patient_context, adherence_history)
        analysis = report.get('adherence_analysis', {})
        print(f"Trend: {analysis.get('trend', 'stable')}")
        print(f"Risk Level / Status: {report.get('overall_status', 'info').upper()}")
        warnings = analysis.get('warnings', [])
        if warnings:
            print(f"⚠️  Warnings: {len(warnings)}")
            for warning in warnings[:2]:  # Show first 2
                print(f"   • {warning.get('message', warning)}")
        
        print("\n" + "─"*70)
        if sys.stdin and sys.stdin.isatty():
            input("\nPress Enter to continue to next patient...\n")
        else:
            print("\nProceeding to next patient...\n")
    
    # Final Summary
    print_header("SYSTEM DEMONSTRATION COMPLETE")
    
    # Calculate overall statistics
    total_records = len(medication_df)
    total_taken = medication_df['taken'].sum()
    overall_adherence = (total_taken / total_records) * 100
    
    print("📊 OVERALL SYSTEM STATISTICS:")
    print(f"   Total Patients: {len(patients)}")
    print(f"   Total Medication Records: {total_records}")
    print(f"   Overall Adherence Rate: {overall_adherence:.1f}%")
    print(f"   Patients Demonstrated: {len(demo_patients)}")
    
    print("\n✅ CAPABILITIES DEMONSTRATED:")
    print("   ✓ Multi-Agent Orchestration (3 specialized agents)")
    print("   ✓ RAG-based Knowledge Retrieval (FAISS + Embeddings)")
    print("   ✓ Drug Interaction Detection")
    print("   ✓ Personalized Medication Information")
    print("   ✓ Adherence Pattern Analysis")
    print("   ✓ AI-Powered Coaching & Reminders")
    print("   ✓ Risk Assessment & Monitoring")
    
    print("\n🎉 All system components functioning successfully!")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
