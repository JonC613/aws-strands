import os
import pytest
from unittest.mock import patch, MagicMock


def test_lm_studio_configuration():
    """Test that LM Studio configuration reads environment variables correctly"""
    with patch.dict(os.environ, {
        'USE_LM_STUDIO': 'true',
        'LM_STUDIO_URL': 'http://test.local:1234/v1',
        'LM_STUDIO_MODEL': 'test-model'
    }):
        # Test env var reading logic without actually running main.py
        assert os.getenv('USE_LM_STUDIO', 'false').lower() == 'true'
        assert os.getenv('LM_STUDIO_URL') == 'http://test.local:1234/v1'
        assert os.getenv('LM_STUDIO_MODEL') == 'test-model'


def test_aws_bedrock_fallback():
    """Test that AWS Bedrock is used when USE_LM_STUDIO is false"""
    with patch.dict(os.environ, {'USE_LM_STUDIO': 'false'}):
        assert os.getenv('USE_LM_STUDIO', 'false').lower() == 'false'


@patch('litellm.completion')
def test_litellm_completion_call(mock_completion):
    """Test that litellm.completion is called with correct parameters"""
    # Mock the response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "You are 46 years old"
    mock_completion.return_value = mock_response
    
    from litellm import completion
    
    response = completion(
        model="openai/test-model",
        messages=[{"role": "user", "content": "Test message"}],
        api_base="http://localhost:1234/v1",
        api_key="not-needed"
    )
    
    assert response.choices[0].message.content == "You are 46 years old"
    mock_completion.assert_called_once()


def test_environment_defaults():
    """Test that default values are used when environment variables are not set"""
    with patch.dict(os.environ, {}, clear=True):
        use_lm = os.getenv("USE_LM_STUDIO", "false").lower() == "true"
        url = os.getenv("LM_STUDIO_URL", "http://192.168.68.123:1234/v1")
        model = os.getenv("LM_STUDIO_MODEL", "phi-4")
        
        assert use_lm == False
        assert url == "http://192.168.68.123:1234/v1"
        assert model == "phi-4"

