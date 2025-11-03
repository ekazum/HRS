"""
Consultation Agent Module for Health Recommendation System.

This module defines the base ConsultationAgent interface and provides
a dummy implementation that can be easily replaced with a real agent.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any


class ConsultationAgent(ABC):
    """Abstract base class for medical consultation agents."""
    
    @abstractmethod
    def consult(self, symptoms: str, duration: str, severity: int, additional_info: str) -> Dict[str, Any]:
        """
        Consult the agent for medical advice.
        
        Args:
            symptoms: Patient's described symptoms
            duration: Duration of symptoms
            severity: Severity rating (1-10)
            additional_info: Additional patient information
            
        Returns:
            Dict containing:
                - diagnosis_suggestions: List of possible diagnoses
                - recommended_tests: List of recommended diagnostic tests
                - urgency_level: Urgency level (low, medium, high, emergency)
        """
        pass


class DummyConsultationAgent(ConsultationAgent):
    """
    Dummy implementation of ConsultationAgent for demonstration.
    
    This agent provides mock responses that can be used for testing
    and as a placeholder before implementing a real medical consultation agent.
    """
    
    def consult(self, symptoms: str, duration: str, severity: int, additional_info: str) -> Dict[str, Any]:
        """
        Provide dummy medical consultation based on severity.
        
        Args:
            symptoms: Patient's described symptoms
            duration: Duration of symptoms
            severity: Severity rating (1-10)
            additional_info: Additional patient information
            
        Returns:
            Dict containing mock diagnosis suggestions, recommended tests, and urgency level
        """
        # Determine urgency based on severity
        if severity >= 8:
            urgency_level = "high"
            diagnosis_suggestions = [
                "Acute condition requiring immediate attention",
                "Possible severe infection or inflammation",
                "Potential emergency situation"
            ]
            recommended_tests = [
                "Complete Blood Count (CBC)",
                "Comprehensive Metabolic Panel",
                "Imaging (X-ray/CT scan) if indicated",
                "Emergency vital signs monitoring"
            ]
        elif severity >= 5:
            urgency_level = "medium"
            diagnosis_suggestions = [
                "Moderate condition requiring medical evaluation",
                "Possible infection or inflammatory process",
                "Common illness with manageable symptoms"
            ]
            recommended_tests = [
                "Complete Blood Count (CBC)",
                "Basic Metabolic Panel",
                "Urinalysis if indicated",
                "Follow-up with primary care physician"
            ]
        else:
            urgency_level = "low"
            diagnosis_suggestions = [
                "Mild condition that may resolve with self-care",
                "Minor illness or temporary discomfort",
                "Low-risk symptoms"
            ]
            recommended_tests = [
                "Monitor symptoms for 24-48 hours",
                "Consider over-the-counter remedies",
                "Schedule routine check-up if symptoms persist",
                "Rest and hydration"
            ]
        
        return {
            "diagnosis_suggestions": diagnosis_suggestions,
            "recommended_tests": recommended_tests,
            "urgency_level": urgency_level
        }


def get_consultation_agent() -> ConsultationAgent:
    """
    Factory function to get the appropriate consultation agent.
    
    This function can be modified to return different agent implementations
    based on configuration or environment variables.
    
    Returns:
        ConsultationAgent: An instance of a consultation agent
    """
    # Check for environment variable to select agent type
    agent_type = os.getenv('CONSULTATION_AGENT_TYPE', 'dummy').lower()
    
    if agent_type == 'dummy':
        return DummyConsultationAgent()
    # Future agent types can be added here:
    # elif agent_type == 'real':
    #     return RealConsultationAgent()
    else:
        # Default to dummy agent
        return DummyConsultationAgent()
