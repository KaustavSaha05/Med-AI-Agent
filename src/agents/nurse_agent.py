"""
Nurse Agent - Provides adherence coaching and patient education.
"""
from typing import Dict, Any, List
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class NurseAgent:
    """Agent for medication adherence coaching and patient education."""
    
    def __init__(self, google_api_key: str = None):
        """Initialize the nurse agent."""
        logger.info("Initializing Nurse Agent...")
        
        google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in .env or pass it explicitly.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.8  # Slightly higher for empathetic responses
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a compassionate and knowledgeable nurse specializing in medication adherence coaching. Your role is to:

1. **Provide Emotional Support**: Be empathetic, understanding, and encouraging
2. **Educate Patients**: Explain why medications are important in simple terms
3. **Offer Practical Tips**: Suggest strategies to remember medications and overcome barriers
4. **Monitor Adherence**: Gently inquire about missed doses without being judgmental
5. **Motivate**: Use positive reinforcement and celebrate small wins

Patient Information:
{patient_info}

Recent Adherence Data:
{adherence_data}

Communication Style:
- Use warm, friendly language
- Ask open-ended questions
- Validate patient concerns
- Offer specific, actionable advice
- Encourage without pressuring
"""),
            ("human", "{message}")
        ])
        
        logger.info("Nurse Agent initialized successfully")
    
    def generate_reminder_message(self, patient_context: Dict, medication: str, 
                                  time_of_day: str = "morning") -> str:
        """
        Generate a personalized medication reminder.
        
        Args:
            patient_context: Patient information
            medication: Medication name
            time_of_day: Time of day (morning, afternoon, evening)
            
        Returns:
            Personalized reminder message
        """
        logger.info(f"Generating reminder for {medication} at {time_of_day}")
        
        patient_name = patient_context.get('name', 'there')
        
        messages = self.prompt_template.format_messages(
            patient_info=f"Name: {patient_name}",
            adherence_data="Generating reminder",
            message=f"Create a friendly, encouraging reminder for {patient_name} to take {medication} this {time_of_day}. Keep it brief (2-3 sentences) and motivating."
        )
        
        response = self.llm.invoke(messages)
        return response.content
    
    def provide_adherence_coaching(self, patient_context: Dict, 
                                   adherence_rate: float, 
                                   missed_doses: List[Dict]) -> Dict[str, Any]:
        """
        Provide personalized adherence coaching based on patient's adherence patterns.
        
        Args:
            patient_context: Patient information
            adherence_rate: Current adherence rate (0-1)
            missed_doses: List of recently missed doses
            
        Returns:
            Coaching response with strategies and encouragement
        """
        logger.info(f"Providing adherence coaching (current rate: {adherence_rate:.1%})")
        
        # Analyze adherence pattern
        pattern_analysis = self._analyze_adherence_pattern(missed_doses)
        
        # Format patient info
        patient_info = f"""
        Name: {patient_context.get('name', 'Patient')}
        Medications: {', '.join(patient_context.get('medications', []))}
        Current Adherence Rate: {adherence_rate:.1%}
        """
        
        # Format adherence data
        adherence_data = f"""
        Missed Doses (last 7 days): {len(missed_doses)}
        Pattern: {pattern_analysis}
        """
        
        # Generate coaching message
        coaching_prompt = f"""
        The patient's adherence rate is {adherence_rate:.1%}. They have missed {len(missed_doses)} doses recently.
        
        Pattern Analysis: {pattern_analysis}
        
        Please provide:
        1. Empathetic acknowledgment of their situation
        2. Specific strategies to improve adherence based on the identified pattern
        3. Encouragement and positive reinforcement
        4. One actionable tip they can implement today
        
        Keep the tone warm and supportive, not preachy.
        """
        
        messages = self.prompt_template.format_messages(
            patient_info=patient_info,
            adherence_data=adherence_data,
            message=coaching_prompt
        )
        
        response = self.llm.invoke(messages)
        
        return {
            'coaching_message': response.content,
            'adherence_rate': adherence_rate,
            'pattern': pattern_analysis,
            'strategies': self._get_adherence_strategies(pattern_analysis)
        }
    
    def _analyze_adherence_pattern(self, missed_doses: List[Dict]) -> str:
        """Analyze pattern in missed doses."""
        if not missed_doses:
            return "Excellent adherence - no missed doses detected"
        
        # Extract times and days
        times = [dose.get('time_of_day', 'unknown') for dose in missed_doses]
        days = [dose.get('day_of_week', 'unknown') for dose in missed_doses]
        
        # Find patterns
        patterns = []
        
        # Time-based pattern
        time_counts = {}
        for time in times:
            time_counts[time] = time_counts.get(time, 0) + 1
        
        if time_counts:
            most_missed_time = max(time_counts, key=time_counts.get)
            if time_counts[most_missed_time] >= len(missed_doses) * 0.5:
                patterns.append(f"Most doses missed in the {most_missed_time}")
        
        # Frequency pattern
        if len(missed_doses) > 5:
            patterns.append("Frequent missed doses - may need system/reminder help")
        elif len(missed_doses) <= 2:
            patterns.append("Occasional missed doses - generally good adherence")
        
        return " | ".join(patterns) if patterns else "Random pattern - no clear trend"
    
    def _get_adherence_strategies(self, pattern: str) -> List[str]:
        """Get specific strategies based on adherence pattern."""
        strategies = []
        
        if "morning" in pattern.lower():
            strategies.extend([
                "Set a morning alarm as a medication reminder",
                "Place medications next to your toothbrush or coffee maker",
                "Take medications before breakfast as part of morning routine"
            ])
        elif "evening" in pattern.lower():
            strategies.extend([
                "Set an evening phone reminder with a pleasant ringtone",
                "Link medication to a nightly activity (e.g., before brushing teeth)",
                "Use a bedside pill organizer"
            ])
        
        if "frequent" in pattern.lower():
            strategies.extend([
                "Use a 7-day pill organizer to track doses visually",
                "Consider a medication reminder app with multiple daily alerts",
                "Ask family members to help remind you"
            ])
        
        # General strategies
        strategies.extend([
            "Keep a medication diary to track progress",
            "Celebrate weekly adherence milestones",
            "Discuss concerns or side effects with your doctor"
        ])
        
        return strategies[:5]  # Return top 5 strategies
    
    def handle_patient_concern(self, patient_context: Dict, concern: str) -> str:
        """
        Address patient concerns about medications with empathy.
        
        Args:
            patient_context: Patient information
            concern: Patient's concern or question
            
        Returns:
            Empathetic response
        """
        logger.info(f"Addressing patient concern: {concern[:50]}...")
        
        patient_info = f"Name: {patient_context.get('name', 'Patient')}"
        
        messages = self.prompt_template.format_messages(
            patient_info=patient_info,
            adherence_data="Patient has expressed a concern",
            message=f"Patient says: '{concern}'\n\nPlease respond with empathy, validate their concern, provide helpful information, and encourage them to discuss with their healthcare provider if needed."
        )
        
        response = self.llm.invoke(messages)
        return response.content

# Example usage
if __name__ == "__main__":
    agent = NurseAgent()
    
    patient_context = {
        'name': 'John Doe',
        'medications': ['Lisinopril', 'Metformin', 'Atorvastatin'],
        'conditions': ['Hypertension', 'Diabetes']
    }
    
    # Test reminder
    print("=== Medication Reminder ===")
    reminder = agent.generate_reminder_message(patient_context, "Lisinopril", "morning")
    print(reminder)
    
    # Test adherence coaching
    print("\n=== Adherence Coaching ===")
    missed_doses = [
        {'medication': 'Metformin', 'time_of_day': 'evening', 'day_of_week': 'Monday'},
        {'medication': 'Metformin', 'time_of_day': 'evening', 'day_of_week': 'Wednesday'},
        {'medication': 'Lisinopril', 'time_of_day': 'morning', 'day_of_week': 'Sunday'}
    ]
    
    coaching = agent.provide_adherence_coaching(patient_context, 0.65, missed_doses)
    print(coaching['coaching_message'])
    print(f"\nPattern: {coaching['pattern']}")
    print("\nSuggested Strategies:")
    for i, strategy in enumerate(coaching['strategies'], 1):
        print(f"{i}. {strategy}")
