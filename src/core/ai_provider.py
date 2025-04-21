"""
AIProvider 基类与 Gemini/OpenAI API 封装
参考学术分类体系项目结构
"""

from typing import Dict, Any, List

class AIProvider:
    """AI提供者基类"""
    def __init__(self, api_config: Dict[str, Any]):
        self.api_config = api_config
        self.current_model = self._get_default_model()

    def _get_default_model(self) -> str:
        if not self.api_config.get('models'):
            raise ValueError("No models configured for this provider")
        return self.api_config['models'][0]['name']

    def set_model(self, model_name: str) -> None:
        for model in self.api_config['models']:
            if model['name'] == model_name:
                self.current_model = model_name
                return
        raise ValueError(f"Model {model_name} not found in configuration")

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError

    def get_available_models(self) -> List[str]:
        return [model['name'] for model in self.api_config['models']]

import yaml
import google.generativeai as genai

class GeminiProvider(AIProvider):
    """Google Gemini API 提供者"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-04-17"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        # Gemini API: system_prompt 可拼接到前面
        full_prompt = system_prompt + "\n" + prompt if system_prompt else prompt
        response = self.model.generate_content(full_prompt)
        return response.text if hasattr(response, "text") else str(response)
