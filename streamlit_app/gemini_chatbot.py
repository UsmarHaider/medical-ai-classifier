"""
Google Gemini AI Chatbot Integration
Medical AI Assistant for explaining classification results
"""
import os
import sys
from typing import Optional, List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import GEMINI_CONFIG


class GeminiChatbot:
    """
    Gemini AI Chatbot for medical image classification assistance.
    """

    SYSTEM_PROMPT = """You are a helpful medical AI assistant specialized in medical imaging analysis.
    You can help users understand medical image classification results, explain medical conditions,
    and provide general health information.

    Important guidelines:
    1. Always remind users that AI predictions are for educational purposes only
    2. Recommend consulting healthcare professionals for actual diagnosis
    3. Explain results in simple, understandable terms
    4. Be empathetic and supportive
    5. Provide factual medical information when asked
    6. Never provide specific treatment recommendations
    7. Encourage users to seek professional medical advice

    You are integrated with a medical image classification system that can detect:
    - Kidney Cancer (Normal, Cyst, Tumor, Stone)
    - Cervical Cancer (Normal, Abnormal)
    - Alzheimer's Disease (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
    - COVID-19 (Normal, COVID, Viral Pneumonia)
    - Pneumonia (Normal, Pneumonia)
    - Tuberculosis (Normal, Tuberculosis)
    - Monkeypox (Normal, Monkeypox)
    - Malaria (Uninfected, Parasitized)
    """

    def __init__(self, api_key: str):
        """
        Initialize the Gemini chatbot.

        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        self.model = None
        self.chat_session = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            # Set up generation config
            generation_config = {
                "temperature": GEMINI_CONFIG.get("temperature", 0.7),
                "max_output_tokens": GEMINI_CONFIG.get("max_output_tokens", 2048),
                "top_p": 0.95,
                "top_k": 40
            }

            # Set up safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]

            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=GEMINI_CONFIG.get("model_name", "gemini-pro"),
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self.SYSTEM_PROMPT
            )

            # Start chat session
            self.chat_session = self.model.start_chat(history=[])

        except ImportError:
            print("google-generativeai package not installed")
            self.model = None
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            self.model = None

    def get_response(self,
                     user_message: str,
                     context: str = None) -> str:
        """
        Get a response from the chatbot.

        Args:
            user_message: User's message
            context: Additional context (e.g., classification results)

        Returns:
            Chatbot response
        """
        if self.model is None:
            return self._get_fallback_response(user_message)

        try:
            # Build message with context
            full_message = user_message
            if context:
                full_message = f"""Context from recent classification:
{context}

User question: {user_message}"""

            # Get response
            response = self.chat_session.send_message(full_message)
            return response.text

        except Exception as e:
            print(f"Error getting Gemini response: {e}")
            return self._get_fallback_response(user_message)

    def _get_fallback_response(self, user_message: str) -> str:
        """Provide fallback responses when Gemini is unavailable."""
        user_message_lower = user_message.lower()

        # Common questions and responses
        responses = {
            "result": """Based on your classification result, I can provide some general information.
However, please remember that AI predictions are for educational purposes only.
For accurate diagnosis and treatment, please consult a healthcare professional.

Would you like me to explain what the different classification categories mean?""",

            "symptom": """I understand you're asking about symptoms. While I can provide general information,
it's important to consult a healthcare provider for proper evaluation.

Common symptoms can vary depending on the condition. Would you like general information
about a specific condition from our classification system?""",

            "treatment": """I cannot provide specific treatment recommendations as this requires
professional medical evaluation.

Please consult with a qualified healthcare provider who can:
1. Review your complete medical history
2. Perform proper diagnostic tests
3. Recommend appropriate treatment options

Is there something else I can help explain?""",

            "accuracy": """Our AI classification system uses state-of-the-art deep learning models
including CNN, VGG16, ResNet50, and Vision Transformers.

However, it's important to note:
1. AI predictions should be validated by medical professionals
2. Accuracy varies depending on image quality and conditions
3. This tool is for educational and screening purposes only

Would you like to know more about how the system works?""",

            "default": """I'm here to help you understand your medical image classification results
and provide general health information.

Please note that I'm an AI assistant and cannot replace professional medical advice.
Always consult healthcare providers for diagnosis and treatment.

How can I assist you today?"""
        }

        # Match response
        for keyword, response in responses.items():
            if keyword in user_message_lower:
                return response

        return responses["default"]

    def explain_condition(self, condition: str) -> str:
        """
        Get an explanation for a medical condition.

        Args:
            condition: Name of the condition

        Returns:
            Explanation text
        """
        prompt = f"""Please explain {condition} in simple terms:
1. What is it?
2. What are common symptoms?
3. How is it typically diagnosed?
4. Why is early detection important?

Remember to recommend consulting healthcare professionals for proper diagnosis."""

        return self.get_response(prompt)

    def explain_classification_result(self,
                                       dataset: str,
                                       predicted_class: str,
                                       confidence: float) -> str:
        """
        Explain a classification result.

        Args:
            dataset: Name of the dataset
            predicted_class: Predicted class name
            confidence: Prediction confidence

        Returns:
            Explanation text
        """
        prompt = f"""A medical image was classified with the following result:
- Type of scan/image: {dataset}
- Classification result: {predicted_class}
- Confidence level: {confidence:.1%}

Please explain:
1. What this result means
2. What factors might affect this prediction
3. What the user should do next
4. Important disclaimers about AI predictions

Keep the explanation clear and accessible to non-medical professionals."""

        return self.get_response(prompt)

    def get_follow_up_questions(self, context: str) -> List[str]:
        """
        Generate relevant follow-up questions based on context.

        Args:
            context: Current conversation context

        Returns:
            List of suggested follow-up questions
        """
        default_questions = [
            "What does my classification result mean?",
            "Should I be concerned about this result?",
            "What should I do next?",
            "How accurate is this AI classification?",
            "What other tests might be needed?"
        ]

        if self.model is None:
            return default_questions

        try:
            prompt = f"""Based on this context:
{context}

Generate 5 relevant follow-up questions that a patient might want to ask.
Return only the questions, one per line, without numbering."""

            response = self.model.generate_content(prompt)
            questions = response.text.strip().split('\n')
            return [q.strip() for q in questions if q.strip()][:5]

        except Exception:
            return default_questions


class MedicalKnowledgeBase:
    """
    Simple knowledge base for medical information.
    Used as fallback when Gemini is unavailable.
    """

    CONDITIONS = {
        "kidney_cancer": {
            "name": "Kidney Cancer",
            "description": "Kidney cancer is cancer that begins in the kidneys.",
            "classes": {
                "Normal": "No abnormalities detected in the kidney tissue.",
                "Cyst": "A fluid-filled sac in the kidney. Most cysts are benign.",
                "Tumor": "An abnormal growth that may be benign or malignant.",
                "Stone": "A hard mineral deposit that forms in the kidney."
            },
            "risk_factors": ["Smoking", "Obesity", "High blood pressure", "Family history"],
            "next_steps": "Consult a urologist for further evaluation and imaging studies."
        },
        "cervical_cancer": {
            "name": "Cervical Cancer",
            "description": "Cancer that occurs in the cells of the cervix.",
            "classes": {
                "Normal": "No abnormal cells detected.",
                "Abnormal": "Abnormal cells detected that may require further testing."
            },
            "risk_factors": ["HPV infection", "Smoking", "Weakened immune system"],
            "next_steps": "Consult a gynecologist for colposcopy or biopsy if needed."
        },
        "alzheimer": {
            "name": "Alzheimer's Disease",
            "description": "A progressive neurological disorder affecting memory and thinking.",
            "classes": {
                "NonDemented": "No signs of dementia detected.",
                "VeryMildDemented": "Very early signs of cognitive decline.",
                "MildDemented": "Mild cognitive impairment present.",
                "ModerateDemented": "Moderate dementia with significant cognitive decline."
            },
            "risk_factors": ["Age", "Family history", "Genetics", "Cardiovascular factors"],
            "next_steps": "Consult a neurologist for comprehensive cognitive assessment."
        },
        "covid19": {
            "name": "COVID-19",
            "description": "Respiratory illness caused by SARS-CoV-2 virus.",
            "classes": {
                "Normal": "No signs of COVID-19 or viral pneumonia.",
                "COVID": "Patterns consistent with COVID-19 infection.",
                "Viral Pneumonia": "Signs of viral pneumonia (non-COVID)."
            },
            "risk_factors": ["Close contact with infected persons", "Crowded settings"],
            "next_steps": "Get PCR test confirmation and consult a healthcare provider."
        },
        "pneumonia": {
            "name": "Pneumonia",
            "description": "Infection that inflames air sacs in one or both lungs.",
            "classes": {
                "Normal": "No signs of pneumonia detected.",
                "Pneumonia": "Signs consistent with pneumonia infection."
            },
            "risk_factors": ["Age (very young or elderly)", "Chronic diseases", "Smoking"],
            "next_steps": "Consult a pulmonologist for proper diagnosis and treatment."
        },
        "tuberculosis": {
            "name": "Tuberculosis",
            "description": "Bacterial infection that mainly affects the lungs.",
            "classes": {
                "Normal": "No signs of tuberculosis detected.",
                "Tuberculosis": "Signs consistent with tuberculosis infection."
            },
            "risk_factors": ["Weakened immune system", "Close contact with TB patients"],
            "next_steps": "Get sputum test and consult an infectious disease specialist."
        },
        "monkeypox": {
            "name": "Monkeypox",
            "description": "Viral disease causing skin lesions and flu-like symptoms.",
            "classes": {
                "Normal": "No signs of monkeypox detected.",
                "Monkeypox": "Skin lesions consistent with monkeypox."
            },
            "risk_factors": ["Close contact with infected persons or animals"],
            "next_steps": "Consult a dermatologist or infectious disease specialist."
        },
        "malaria": {
            "name": "Malaria",
            "description": "Parasitic disease transmitted by mosquitoes.",
            "classes": {
                "Uninfected": "No malaria parasites detected in blood cells.",
                "Parasitized": "Malaria parasites detected in blood cells."
            },
            "risk_factors": ["Travel to endemic areas", "Mosquito exposure"],
            "next_steps": "Consult a healthcare provider immediately for treatment."
        }
    }

    @classmethod
    def get_condition_info(cls, condition_key: str) -> Optional[Dict]:
        """Get information about a condition."""
        return cls.CONDITIONS.get(condition_key)

    @classmethod
    def explain_class(cls, condition_key: str, class_name: str) -> str:
        """Explain a specific class for a condition."""
        condition = cls.CONDITIONS.get(condition_key)
        if condition:
            return condition["classes"].get(class_name, "No information available.")
        return "Condition not found in knowledge base."
