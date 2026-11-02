"""
Monitor Agent - Provides early warnings for non-adherence and health risks.
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MonitorAgent:
    """Agent for monitoring medication adherence and providing early warnings."""
    
    def __init__(self):
        """Initialize the monitor agent."""
        logger.info("Initializing Monitor Agent...")
        
        # Warning thresholds
        self.thresholds = {
            'critical_adherence': 0.50,  # Below 50% is critical
            'low_adherence': 0.70,        # Below 70% is concerning
            'consecutive_missed': 3,       # 3 consecutive missed doses
            'weekly_missed_threshold': 5   # 5+ missed doses in a week
        }
        
        logger.info(" Monitor Agent initialized successfully")
    
    def analyze_adherence_trend(self, adherence_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze adherence trends over time.
        
        Args:
            adherence_history: List of adherence records with dates and taken status
            
        Returns:
            Trend analysis with warnings
        """
        logger.info(f"Analyzing adherence trend for {len(adherence_history)} records")
        
        if not adherence_history:
            return {
                'trend': 'insufficient_data',
                'warnings': [],
                'severity': 'info'
            }
        
        # Calculate adherence rates for different periods
        total_doses = len(adherence_history)
        taken_doses = sum(1 for record in adherence_history if record.get('taken', False))
        overall_rate = taken_doses / total_doses if total_doses > 0 else 0
        
        # Last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_records = [
            r for r in adherence_history 
            if datetime.fromisoformat(r.get('scheduled_time', '2000-01-01')) > seven_days_ago
        ]
        recent_rate = sum(1 for r in recent_records if r.get('taken', False)) / len(recent_records) if recent_records else 0
        
        # Detect trend
        trend = self._determine_trend(overall_rate, recent_rate)
        
        # Generate warnings
        warnings = self._generate_warnings(adherence_history, overall_rate, recent_rate)
        
        # Determine severity
        severity = self._determine_severity(warnings, overall_rate)
        
        return {
            'trend': trend,
            'overall_adherence': overall_rate,
            'recent_adherence': recent_rate,
            'warnings': warnings,
            'severity': severity,
            'total_doses': total_doses,
            'missed_doses': total_doses - taken_doses
        }
    
    def _determine_trend(self, overall_rate: float, recent_rate: float) -> str:
        """Determine adherence trend."""
        if recent_rate > overall_rate + 0.1:
            return "improving"
        elif recent_rate < overall_rate - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_warnings(self, adherence_history: List[Dict], 
                          overall_rate: float, recent_rate: float) -> List[Dict]:
        """Generate warnings based on adherence patterns."""
        warnings = []
        
        # Critical adherence warning
        if overall_rate < self.thresholds['critical_adherence']:
            warnings.append({
                'type': 'critical_adherence',
                'message': f" CRITICAL: Overall adherence is {overall_rate:.1%}, well below the recommended 80%",
                'severity': 'critical',
                'action': "Immediate intervention required. Consider medication therapy management consultation."
            })
        elif overall_rate < self.thresholds['low_adherence']:
            warnings.append({
                'type': 'low_adherence',
                'message': f" WARNING: Overall adherence is {overall_rate:.1%}, below the target of 80%",
                'severity': 'warning',
                'action': "Implement adherence support strategies and identify barriers."
            })
        
        # Declining trend warning
        if recent_rate < overall_rate - 0.15:
            warnings.append({
                'type': 'declining_trend',
                'message': f"ALERT: Adherence has declined recently from {overall_rate:.1%} to {recent_rate:.1%}",
                'severity': 'warning',
                'action': "Investigate recent changes or barriers. Patient may need additional support."
            })
        
        # Consecutive missed doses
        consecutive_missed = self._find_consecutive_missed(adherence_history)
        if consecutive_missed >= self.thresholds['consecutive_missed']:
            warnings.append({
                'type': 'consecutive_missed',
                'message': f"ALERT: {consecutive_missed} consecutive doses missed",
                'severity': 'warning',
                'action': "Contact patient immediately to assess situation and provide support."
            })
        
        # Pattern-based warnings
        pattern_warnings = self._detect_patterns(adherence_history)
        warnings.extend(pattern_warnings)
        
        return warnings
    
    def _find_consecutive_missed(self, adherence_history: List[Dict]) -> int:
        """Find maximum consecutive missed doses."""
        max_consecutive = 0
        current_consecutive = 0
        
        # Sort by date
        sorted_history = sorted(
            adherence_history,
            key=lambda x: datetime.fromisoformat(x.get('scheduled_time', '2000-01-01'))
        )
        
        for record in sorted_history:
            if not record.get('taken', False):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _detect_patterns(self, adherence_history: List[Dict]) -> List[Dict]:
        """Detect patterns in missed doses."""
        warnings = []
        
        # Time-of-day pattern
        missed_by_time = {'morning': 0, 'afternoon': 0, 'evening': 0}
        
        for record in adherence_history:
            if not record.get('taken', False):
                time_str = record.get('scheduled_time', '')
                try:
                    hour = datetime.fromisoformat(time_str).hour
                    if hour < 12:
                        missed_by_time['morning'] += 1
                    elif hour < 17:
                        missed_by_time['afternoon'] += 1
                    else:
                        missed_by_time['evening'] += 1
                except:
                    pass
        
        total_missed = sum(missed_by_time.values())
        if total_missed > 0:
            for time_of_day, count in missed_by_time.items():
                if count / total_missed > 0.6:  # 60% of missed doses in one period
                    warnings.append({
                        'type': 'time_pattern',
                        'message': f" PATTERN: Most missed doses occur in the {time_of_day}",
                        'severity': 'info',
                        'action': f"Consider setting specific reminders for {time_of_day} medications."
                    })
        
        return warnings
    
    def _determine_severity(self, warnings: List[Dict], adherence_rate: float) -> str:
        """Determine overall severity level."""
        if any(w['severity'] == 'critical' for w in warnings):
            return 'critical'
        elif any(w['severity'] == 'warning' for w in warnings):
            return 'warning'
        elif adherence_rate >= 0.80:
            return 'good'
        else:
            return 'info'
    
    def check_medication_interactions(self, medications: List[str]) -> List[Dict]:
        """
        Check for potential drug interactions in medication list.
        
        Args:
            medications: List of medication names
            
        Returns:
            List of interaction warnings
        """
        logger.info(f"Checking interactions for {len(medications)} medications")
        
        # Known interaction pairs (simplified)
        known_interactions = {
            ('Lisinopril', 'Ibuprofen'): {
                'severity': 'moderate',
                'description': 'NSAIDs may reduce effectiveness of ACE inhibitors',
                'recommendation': 'Monitor blood pressure and consider alternative pain relief'
            },
            ('Warfarin', 'Aspirin'): {
                'severity': 'high',
                'description': 'Increased bleeding risk',
                'recommendation': 'Use together only under close medical supervision'
            },
            ('Metformin', 'Alcohol'): {
                'severity': 'moderate',
                'description': 'Increased risk of lactic acidosis',
                'recommendation': 'Limit alcohol consumption'
            },
            ('Sertraline', 'Alprazolam'): {
                'severity': 'moderate',
                'description': 'Increased sedation and respiratory depression',
                'recommendation': 'Use caution when driving or operating machinery'
            }
        }
        
        interactions = []
        
        # Check all pairs
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                # Check both orderings
                for pair in [(med1, med2), (med2, med1)]:
                    if pair in known_interactions:
                        interaction = known_interactions[pair]
                        interactions.append({
                            'medications': [med1, med2],
                            'severity': interaction['severity'],
                            'description': interaction['description'],
                            'recommendation': interaction['recommendation']
                        })
        
        logger.info(f"Found {len(interactions)} potential interactions")
        return interactions
    
    def generate_monitoring_report(self, patient_context: Dict, 
                                   adherence_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report for a patient.
        
        Args:
            patient_context: Patient information
            adherence_history: Adherence records
            
        Returns:
            Comprehensive monitoring report
        """
        logger.info(f"Generating monitoring report for patient {patient_context.get('patient_id', 'unknown')}")
        
        # Analyze adherence
        adherence_analysis = self.analyze_adherence_trend(adherence_history)
        
        # Check interactions
        medications = patient_context.get('medications', [])
        if isinstance(medications, list) and medications and isinstance(medications[0], dict):
            med_names = [m['name'] for m in medications]
        else:
            med_names = medications
        
        interactions = self.check_medication_interactions(med_names)
        
        # Generate report
        report = {
            'patient_id': patient_context.get('patient_id'),
            'report_date': datetime.now().isoformat(),
            'adherence_analysis': adherence_analysis,
            'drug_interactions': interactions,
            'overall_status': adherence_analysis['severity'],
            'recommendations': self._generate_recommendations(adherence_analysis, interactions)
        }
        
        logger.info(" Monitoring report generated successfully")
        return report
    
    def _generate_recommendations(self, adherence_analysis: Dict, 
                                 interactions: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Adherence-based recommendations
        if adherence_analysis['severity'] == 'critical':
            recommendations.append("Urgent: Schedule medication therapy management consultation")
            recommendations.append("Consider supervised medication administration")
        elif adherence_analysis['severity'] == 'warning':
            recommendations.append("Implement adherence support program")
            recommendations.append("Address identified barriers to adherence")
        
        # Trend-based recommendations
        if adherence_analysis['trend'] == 'declining':
            recommendations.append("Investigate recent changes in patient's life or health status")
        
        # Interaction-based recommendations
        high_risk_interactions = [i for i in interactions if i['severity'] == 'high']
        if high_risk_interactions:
            recommendations.append("Urgent: Review high-risk drug interactions with prescriber")
        
        # Warning-based recommendations
        for warning in adherence_analysis.get('warnings', []):
            if 'action' in warning:
                recommendations.append(warning['action'])
        
        return list(set(recommendations))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    agent = MonitorAgent()
    
    # Example adherence history
    adherence_history = []
    base_date = datetime.now() - timedelta(days=14)
    
    for day in range(14):
        for dose in range(2):  # 2 doses per day
            scheduled_time = base_date + timedelta(days=day, hours=8 + dose*12)
            # Simulate declining adherence
            taken = True if (day < 7 or dose == 0) and (day % 3 != 0) else False
            
            adherence_history.append({
                'scheduled_time': scheduled_time.isoformat(),
                'taken': taken,
                'medication': 'Lisinopril'
            })
    
    patient_context = {
        'patient_id': 'P0001',
        'name': 'John Doe',
        'medications': ['Lisinopril', 'Ibuprofen', 'Metformin']
    }
    
    # Generate report
    report = agent.generate_monitoring_report(patient_context, adherence_history)
    
    print("=== MONITORING REPORT ===\n")
    print(f"Patient: {report['patient_id']}")
    print(f"Overall Status: {report['overall_status'].upper()}\n")
    
    print(f"Adherence: {report['adherence_analysis']['overall_adherence']:.1%}")
    print(f"Trend: {report['adherence_analysis']['trend']}\n")
    
    if report['adherence_analysis']['warnings']:
        print(" WARNINGS:")
        for warning in report['adherence_analysis']['warnings']:
            print(f"  - {warning['message']}")
        print()
    
    if report['drug_interactions']:
        print(" DRUG INTERACTIONS:")
        for interaction in report['drug_interactions']:
            print(f"  - {' + '.join(interaction['medications'])}: {interaction['description']}")
        print()
    
    print(" RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
