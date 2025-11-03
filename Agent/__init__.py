"""
Agent Module for Health Recommendation System.

This module contains medical consultation agents that can provide
diagnostic advice and suggest next tests.
"""

from .consultation_agent import ConsultationAgent, get_consultation_agent

__all__ = ['ConsultationAgent', 'get_consultation_agent']
