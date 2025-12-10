"""
Unit tests for consultation agent module.
"""

import unittest
import os
from unittest.mock import patch
from Agent.consultation_agent import (
    ConsultationAgent,
    DummyConsultationAgent,
    get_consultation_agent
)


class TestDummyConsultationAgent(unittest.TestCase):
    """Tests for DummyConsultationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = DummyConsultationAgent()
    
    def test_consult_high_severity(self):
        """Test consultation with high severity (8+)."""
        result = self.agent.consult(
            symptoms="Severe chest pain and difficulty breathing",
            duration="2 hours",
            severity=9,
            additional_info="History of heart disease"
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['urgency_level'], 'high')
        self.assertIn('diagnosis_suggestions', result)
        self.assertIn('recommended_tests', result)
        self.assertIsInstance(result['diagnosis_suggestions'], list)
        self.assertIsInstance(result['recommended_tests'], list)
        self.assertGreater(len(result['diagnosis_suggestions']), 0)
        self.assertGreater(len(result['recommended_tests']), 0)
    
    def test_consult_medium_severity(self):
        """Test consultation with medium severity (5-7)."""
        result = self.agent.consult(
            symptoms="Persistent cough and fever",
            duration="3 days",
            severity=6,
            additional_info=""
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['urgency_level'], 'medium')
        self.assertIn('diagnosis_suggestions', result)
        self.assertIn('recommended_tests', result)
        self.assertIsInstance(result['diagnosis_suggestions'], list)
        self.assertIsInstance(result['recommended_tests'], list)
    
    def test_consult_low_severity(self):
        """Test consultation with low severity (1-4)."""
        result = self.agent.consult(
            symptoms="Mild headache",
            duration="1 day",
            severity=3,
            additional_info=""
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['urgency_level'], 'low')
        self.assertIn('diagnosis_suggestions', result)
        self.assertIn('recommended_tests', result)
        self.assertIsInstance(result['diagnosis_suggestions'], list)
        self.assertIsInstance(result['recommended_tests'], list)
    
    def test_consult_boundary_severity_high(self):
        """Test consultation at boundary severity (8)."""
        result = self.agent.consult(
            symptoms="Test symptoms",
            duration="1 day",
            severity=8,
            additional_info=""
        )
        
        self.assertEqual(result['urgency_level'], 'high')
    
    def test_consult_boundary_severity_medium(self):
        """Test consultation at boundary severity (5)."""
        result = self.agent.consult(
            symptoms="Test symptoms",
            duration="1 day",
            severity=5,
            additional_info=""
        )
        
        self.assertEqual(result['urgency_level'], 'medium')
    
    def test_consult_returns_all_required_fields(self):
        """Test that consultation returns all required fields."""
        result = self.agent.consult(
            symptoms="Test symptoms",
            duration="1 day",
            severity=5,
            additional_info=""
        )
        
        self.assertIn('diagnosis_suggestions', result)
        self.assertIn('recommended_tests', result)
        self.assertIn('urgency_level', result)
        self.assertIn(result['urgency_level'], ['low', 'medium', 'high', 'emergency'])


class TestGetConsultationAgent(unittest.TestCase):
    """Tests for get_consultation_agent factory function."""
    
    def test_default_returns_dummy_agent(self):
        """Test that default returns DummyConsultationAgent."""
        with patch.dict(os.environ, {}, clear=False):
            if 'CONSULTATION_AGENT_TYPE' in os.environ:
                del os.environ['CONSULTATION_AGENT_TYPE']
            agent = get_consultation_agent()
            self.assertIsInstance(agent, DummyConsultationAgent)
    
    def test_explicit_dummy_agent(self):
        """Test explicit selection of dummy agent."""
        with patch.dict(os.environ, {'CONSULTATION_AGENT_TYPE': 'dummy'}):
            agent = get_consultation_agent()
            self.assertIsInstance(agent, DummyConsultationAgent)
    
    def test_unknown_agent_type_defaults_to_dummy(self):
        """Test that unknown agent type defaults to dummy."""
        with patch.dict(os.environ, {'CONSULTATION_AGENT_TYPE': 'unknown'}):
            agent = get_consultation_agent()
            self.assertIsInstance(agent, DummyConsultationAgent)
    
    def test_agent_is_consultation_agent(self):
        """Test that returned agent is a ConsultationAgent."""
        agent = get_consultation_agent()
        self.assertIsInstance(agent, ConsultationAgent)
    
    def test_case_insensitive_agent_type(self):
        """Test that agent type is case insensitive."""
        with patch.dict(os.environ, {'CONSULTATION_AGENT_TYPE': 'DUMMY'}):
            agent = get_consultation_agent()
            self.assertIsInstance(agent, DummyConsultationAgent)


class TestConsultationAgentInterface(unittest.TestCase):
    """Tests for ConsultationAgent interface."""
    
    def test_dummy_agent_implements_interface(self):
        """Test that DummyConsultationAgent implements ConsultationAgent."""
        agent = DummyConsultationAgent()
        self.assertIsInstance(agent, ConsultationAgent)
        self.assertTrue(hasattr(agent, 'consult'))
        self.assertTrue(callable(getattr(agent, 'consult')))


if __name__ == '__main__':
    unittest.main()
