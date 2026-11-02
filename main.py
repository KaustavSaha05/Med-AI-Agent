"""
Main entry point for the Agentic AI Medication Management System.
Coordinates multiple agents for comprehensive medication management.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd
from dotenv import load_dotenv
import os
import time
from functools import wraps

sys.path.append(str(Path(__file__).parent))
from src.agents.medication_manager import MedicationManagerAgent
from src.agents.nurse_agent import NurseAgent
from src.agents.monitor_agent import MonitorAgent
from src.utils.config import SYNTHETIC_PATIENTS_DIR
from src.utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()

logger = setup_logger(__name__, "main.log")

def timing(logger):
    """A decorator to log the execution time of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"PERFORMANCE: {func.__name__} took {elapsed_time:.4f} seconds to execute.")
            return result
        return wrapper
    return decorator

class MedicationManagementSystem:
    """Multi-agent system for comprehensive medication management."""
    
    def __init__(self, google_api_key: str = None):
        """Initialize all agents."""
        logger.info("="*60)
        logger.info("Initializing Agentic AI Medication Management System")
        logger.info("="*60)
        
        if not google_api_key:
            raise ValueError("Google API key is required for agent initialization.")
            
        self.medication_manager = MedicationManagerAgent(google_api_key=google_api_key)
        self.nurse_agent = NurseAgent(google_api_key=google_api_key)
        self.monitor_agent = MonitorAgent()
        
        logger.info("All agents initialized successfully\n")
    
    def load_patient(self, patient_id: str) -> Dict[str, Any]:
        """Load patient data from synthetic dataset."""
        try:
            with open(SYNTHETIC_PATIENTS_DIR / "patients.json", 'r') as f:
                patients = json.load(f)
            
            for patient in patients:
                if patient['patient_id'] == patient_id:
                    logger.info(f"Loaded patient: {patient['name']} ({patient_id})")
                    return patient
            
            logger.warning(f"Patient {patient_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading patient: {e}")
            return None
    
    def load_patient_schedule(self, patient_id: str) -> pd.DataFrame:
        """Load medication schedule for a patient."""
        schedule_path = SYNTHETIC_PATIENTS_DIR / "medication_schedules.csv"
        if not schedule_path.exists():
            logger.warning("Medication schedule file not found")
            return pd.DataFrame()

        try:
            df = pd.read_csv(schedule_path, parse_dates=["scheduled_time", "taken_time"], infer_datetime_format=True)
            patient_schedule = df[df["patient_id"] == patient_id].copy()
            if patient_schedule.empty:
                logger.info(f"No schedule records for patient {patient_id}")
            return patient_schedule
        except Exception as e:
            logger.error(f"Error loading medication schedule: {e}")
            return pd.DataFrame()

    @staticmethod
    def _infer_time_of_day(timestamp: pd.Timestamp) -> str:
        """Infer time of day label from timestamp."""
        if pd.isna(timestamp):
            return "unknown"
        hour = timestamp.hour
        if hour < 12:
            return "morning"
        if hour < 17:
            return "afternoon"
        return "evening"

    @timing(logger)
    def generate_patient_report(self, patient_id: str) -> Dict[str, Any]:
        """Generate structured report data for a patient."""
        patient = self.load_patient(patient_id)
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")

        medications = [med['name'] for med in patient.get('medications', [])]
        patient_context = {
            'patient_id': patient.get('patient_id'),
            'name': patient.get('name'),
            'age': patient.get('age'),
            'gender': patient.get('gender'),
            'medications': medications,
            'conditions': patient.get('conditions', []),
            'allergies': patient.get('allergies', [])
        }

        schedule_df = self.load_patient_schedule(patient_id)
        adherence_rate = patient.get('baseline_adherence', 0.0)
        missed_doses: List[Dict[str, Any]] = []
        adherence_history: List[Dict[str, Any]] = []

        if not schedule_df.empty:
            has_taken = 'taken' in schedule_df.columns
            if has_taken:
                adherence_rate = float(schedule_df['taken'].mean())

            adherence_history = [
                {
                    'scheduled_time': row['scheduled_time'].isoformat() if not pd.isna(row['scheduled_time']) else '',
                    'taken': bool(row['taken']) if has_taken and not pd.isna(row['taken']) else False,
                    'medication': row['medication']
                }
                for _, row in schedule_df.iterrows()
            ]

            if has_taken:
                recent_missed = schedule_df[schedule_df['taken'] == False].sort_values('scheduled_time').tail(7)
                missed_doses = [
                    {
                        'medication': row['medication'],
                        'time_of_day': self._infer_time_of_day(row['scheduled_time']),
                        'day_of_week': row['scheduled_time'].strftime('%A') if not pd.isna(row['scheduled_time']) else 'Unknown'
                    }
                    for _, row in recent_missed.iterrows()
                ]

        interactions = self.monitor_agent.check_medication_interactions(medications)

        # Run LLM-dependent tasks sequentially
        medication_query = self.medication_manager.process_query(
            "What should I know about my current medications?",
            patient_context
        )
        time.sleep(2)  # Add a delay between calls

        reminder = self.nurse_agent.generate_reminder_message(
            patient_context,
            medications[0] if medications else "medication",
            "morning"
        )
        time.sleep(2)

        coaching = self.nurse_agent.provide_adherence_coaching(
            patient_context,
            adherence_rate or patient.get('baseline_adherence', 0.0),
            missed_doses
        )
        time.sleep(2)

        monitoring_report = self.monitor_agent.generate_monitoring_report(
            patient_context,
            adherence_history
        )

        schedule_summary = {
            'total_doses': len(schedule_df),
            'taken_doses': int(schedule_df['taken'].sum()) if not schedule_df.empty and 'taken' in schedule_df.columns else 0,
            'adherence_rate': adherence_rate
        }

        return {
            'patient': patient_context,
            'medications_details': patient.get('medications', []),
            'interactions': interactions,
            'medication_query': medication_query,
            'reminder': reminder,
            'coaching': coaching,
            'monitoring': monitoring_report,
            'schedule_summary': schedule_summary,
            'missed_doses': missed_doses
        }

    @timing(logger)
    def comprehensive_patient_check(self, patient_id: str):
        """Run comprehensive check for a patient using all agents."""
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPREHENSIVE PATIENT CHECK: {patient_id}")
        logger.info(f"{'='*60}\n")
        
        try:
            report = self.generate_patient_report(patient_id)
        except ValueError:
            print(f"Patient {patient_id} not found")
            return
        
        patient = report['patient']
        medications = patient.get('medications', [])

        print(f"\n**PATIENT PROFILE**")
        print(f"   Name: {patient.get('name')}")
        print(f"   Age: {patient.get('age')} | Gender: {patient.get('gender')}")
        print(f"   Conditions: {', '.join(patient.get('conditions', []))}")
        print(f"   Medications: {', '.join(medications) if medications else 'None'}")
        allergies = patient.get('allergies', [])
        print(f"   Allergies: {', '.join(allergies) if allergies else 'None'}")
        
        # 1. Check drug interactions (Monitor Agent)
        print(f"\n**DRUG INTERACTION CHECK** (Monitor Agent)")
        print("   " + "-"*50)
        interactions = report['interactions']
        if interactions:
            for interaction in interactions:
                print(f"   {' + '.join(interaction['medications'])}")
                print(f"       Severity: {interaction['severity'].upper()}")
                print(f"       Issue: {interaction['description']}")
                print(f"       Action: {interaction['recommendation']}\n")
        else:
            print("   No known drug interactions detected\n")
        
        # 2. Answer medication question (Medication Manager Agent)
        print(f"\n**MEDICATION QUERY** (Medication Manager Agent)")
        print("   " + "-"*50)
        query = "What should I know about my current medications?"
        print(f"   Q: {query}\n")
        result = report['medication_query']
        print(f"   A: {result['answer']}\n")
        
        if result['warnings']:
            print("   Additional Warnings:")
            for warning in result['warnings']:
                print(f"       {warning}")
            print()
        
        # 3. Generate medication reminder (Nurse Agent)
        print(f"\n**MEDICATION REMINDER** (Nurse Agent)")
        print("   " + "-"*50)
        if medications:
            print(f"   {report['reminder']}\n")
        
        # 4. Adherence coaching (Nurse Agent)
        print(f"\n**ADHERENCE COACHING** (Nurse Agent)")
        print("   " + "-"*50)
        coaching = report['coaching']
        adherence_rate = coaching.get('adherence_rate', patient.get('baseline_adherence', 0.0))
        print(f"   Current Adherence: {adherence_rate:.1%}")
        print(f"   Pattern: {coaching.get('pattern', 'Unknown')}\n")
        print(f"   Coaching Message:\n")
        print(f"   {coaching.get('coaching_message', 'No message generated.')}\n")

        if coaching.get('strategies'):
            print("   Suggested Strategies:")
            for strategy in coaching['strategies']:
                print(f"     • {strategy}")
            print()

        schedule_summary = report.get('schedule_summary', {})
        if schedule_summary:
            print("   Schedule Summary:")
            print(f"     • Total doses: {schedule_summary.get('total_doses', 0)}")
            print(f"     • Doses taken: {schedule_summary.get('taken_doses', 0)}")
            print(f"     • Calculated adherence: {schedule_summary.get('adherence_rate', 0.0):.1%}")
            print()
        
        print(f"\n{'='*60}")
        logger.info("Comprehensive check completed\n")
    
    def interactive_mode(self):
        """Run interactive mode for patient queries."""
        print("\n" + "="*60)
        print("AGENTIC AI MEDICATION MANAGEMENT SYSTEM")
        print("="*60)
        print("\nCommands:")
        print("  - check <patient_id>  : Run comprehensive patient check")
        print("  - ask <question>      : Ask medication question")
        print("  - remind <med>        : Get medication reminder")
        print("  - list                : List available patients")
        print("  - quit                : Exit system")
        print("="*60 + "\n")
        
        # Load default patient for queries
        default_patient = self.load_patient("P0001")
        if default_patient:
            medications = [med['name'] for med in default_patient['medications']]
            current_context = {
                'patient_id': default_patient['patient_id'],
                'name': default_patient['name'],
                'medications': medications,
                'conditions': default_patient['conditions'],
                'allergies': default_patient['allergies']
            }
            print(f"Current Patient: {default_patient['name']} ({default_patient['patient_id']})\n")
        else:
            current_context = {}
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'list':
                    with open(SYNTHETIC_PATIENTS_DIR / "patients.json", 'r') as f:
                        patients = json.load(f)
                    print(f"\nAvailable Patients ({len(patients)} total):")
                    for i, p in enumerate(patients[:10], 1):
                        print(f"   {i}. {p['patient_id']} - {p['name']} ({p['age']} yrs, {len(p['medications'])} meds)")
                    if len(patients) > 10:
                        print(f"   ... and {len(patients) - 10} more")
                    print()
                
                elif user_input.lower().startswith('check '):
                    patient_id = user_input.split(' ', 1)[1].strip()
                    self.comprehensive_patient_check(patient_id)
                
                elif user_input.lower().startswith('ask '):
                    question = user_input.split(' ', 1)[1].strip()
                    result = self.medication_manager.process_query(question, current_context)
                    print(f"\nAnswer:\n{result['answer']}\n")
                    if result['warnings']:
                        print("Warnings:")
                        for warning in result['warnings']:
                            print(f"   {warning}")
                        print()
                
                elif user_input.lower().startswith('remind '):
                    med = user_input.split(' ', 1)[1].strip()
                    reminder = self.nurse_agent.generate_reminder_message(current_context, med)
                    print(f"\n{reminder}\n")
                
                else:
                    print("Unknown command. Type 'quit' to exit or try: check P0001, ask <question>, remind <medication>\n")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
                logger.error(f"Error in interactive mode: {e}", exc_info=True)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI Medication Management System")
    parser.add_argument('--mode', choices=['interactive', 'check'], default='interactive',
                      help='Run mode: interactive or check specific patient')
    parser.add_argument('--patient', type=str, help='Patient ID for check mode')
    
    args = parser.parse_args()
    
    # Initialize system
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please create a .env file and add your key.")
        return
        
    system = MedicationManagementSystem(google_api_key=google_api_key)
    
    if args.mode == 'check':
        if not args.patient:
            print("Error: --patient required for check mode")
            print("Example: python main.py --mode check --patient P0001")
            return
        system.comprehensive_patient_check(args.patient)
    else:
        system.interactive_mode()

if __name__ == "__main__":
    main()
