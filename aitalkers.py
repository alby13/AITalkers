import sys
import os

# Streamlit is the Graphical Interface
# We check here if Streamlit is installed
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    script_name = os.path.basename(__file__)
    print("")
    print("Error: This program requires Streamlit to run.")
    print("Please install Streamlit by running the following command:")
    print("    pip install streamlit")
    print(f"After installing, run this program using:")
    print(f"    streamlit run {script_name}")
    exit(1)

# Check if the script is being run with 'streamlit run'
ctx = get_script_run_ctx()
if ctx is None:
    script_name = os.path.basename(__file__)
    print("")
    print("Error: This program must be run with Streamlit.")
    print(f"Please use the following command to run it:")
    print(f"    streamlit run {script_name}")
    print("Running it with 'python' directly is not supported.")
    exit(1)

# IMPORTS

import streamlit as st
import time
import json
import requests
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
# Gemini GenAI Import
try:
    from google import genai
except ImportError:
    pass  # Just in case that Gemini API isn't available

# INITIALIZATIONS

# Load environment variables (for API keys)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# HELPER FUNCTIONALITY

def api_call_with_retry(url: str, headers: Dict[str, str], payload: Dict, max_retries: int = 3, delay: int = 2) -> Dict:
    """Helper function to call an API with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()  # Raise error for bad responses
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error("API request failed on attempt %d: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                # If the final attempt fails, re-raise the exception to be handled by caller.
                raise e

class AIAgent:
    """Base class for AI agents with full conversation history and context truncation for API calls."""
    def __init__(self, name: str, api_key: str, provider: str):
        self.name = name
        self.api_key = api_key
        self.provider = provider
        self.full_history = []  # Store the complete conversation for user review
        self.max_context_messages = 10  # Maximum number of messages to send as context per API call
    
    def update_history(self, role: str, content: str):
        """Append a message to the full conversation history."""
        self.full_history.append({"role": role, "content": content})
    
    def get_recent_context(self) -> list:
        """
        Retrieve only the most recent messages up to a fixed limit.
        This truncated context is used to stay within API token limits.
        """
        return self.full_history[-self.max_context_messages:]
    
    def generate_response(self, prompt: str, additional_context: list = None) -> str:
        """
        Generate a response using only the recent context.
        The full conversation history is preserved for review.
        """
        # Retrieve recent messages to include as context for the API call.
        effective_context = self.get_recent_context().copy()
        if additional_context:
            effective_context.extend(additional_context)
        
        # Update full history with the user prompt before generating the response.
        self.update_history("user", prompt)
        
        # Here you'd integrate your API call using the 'effective_context' along with the prompt.
        # For demonstration, we can simulate the API response here.
        response = f"Response from {self.name} regarding: {prompt[:50]}..."
        
        # Update full history with the agent's response.
        self.update_history("assistant", response)
        return response

class OpenAIAgent(AIAgent):
    """OpenAI-specific implementation with robust error handling."""
    def generate_response(self, prompt: str, context: List[Dict] = None) -> str:
        """
        Implement the OpenAI API call with error handling.
        In production, replace the mock implementation with an actual API call.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "system", "content": f"You are {self.name}, an AI assistant engaged in a debate. Be logical and critical in your analysis."}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.7,
        }
        
        api_url = "https://api.openai.com/v1/chat/completions"
        
        try:
            # Make the API call with retries.
            api_response = api_call_with_retry(api_url, headers, payload)
            response_text = api_response["choices"][0]["message"]["content"]
        except Exception as e:
            # Log and show a user-friendly error message.
            logging.error("OpenAI API error: %s", e)
            st.error("An error occurred while generating a response from OpenAI. Please try again later.")
            response_text = "Error: Unable to generate response from OpenAI API."
        
        self.update_history("user", prompt)
        self.update_history("assistant", response_text)
        return response_text

class AnthropicAgent(AIAgent):
    """Anthropic-specific implementation with robust error handling."""
    def generate_response(self, prompt: str, context: List[Dict] = None) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": f"You are {self.name}, an AI assistant engaged in a debate. Be logical and critical."}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": "claude-3.5-sonnet",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }
        api_url = "https://api.anthropic.com/v1/messages"
        try:
            response = api_call_with_retry(api_url, headers, payload)
            return response["content"][0]["text"]
        except Exception as e:
            logging.error("Anthropic API error: %s", e)
            st.error("Anthropic API failed.")
            return "Error: Unable to generate response."

class GeminiAgent(AIAgent):
    """Google Gemini-specific implementation with robust error handling."""
    def generate_response(self, prompt: str, context: List[Dict] = None) -> str:
        try:
            client = genai.Client(api_key=self.api_key)
            formatted_prompt = f"You are {self.name}, an AI assistant engaged in a debate. Be logical and critical in your analysis.\n\n"
            if context:
                for message in context:
                    role = message["role"]
                    content = message["content"]
                    formatted_prompt += f"{role.capitalize()}: {content}\n\n"
            formatted_prompt += f"User: {prompt}"
            response = client.models.generate_content(
                model='gemini-2.0-pro-exp-02-05',
                contents=formatted_prompt
            )
            response_text = response.text
        except Exception as e:
            logging.error("Gemini API error: %s", e)
            st.error("Gemini API failed.")
            response_text = "Error: Unable to generate response."
        self.update_history("user", prompt)
        self.update_history("assistant", response_text)
        return response_text

class ModeratorAgent(AIAgent):
    """Moderator agent that determines when consensus is reached with an LLM-based analysis and robust error handling."""
    def check_consensus(self, agent1_final: str, agent2_final: str) -> Tuple[bool, str]:
        """
        Enhanced consensus checking logic:
        Analyze two final answers using the moderator's API call.
        Returns a tuple of (consensus_reached, explanation).
        """
        analysis_prompt = (
            "You are a moderator AI tasked with analyzing two final answers from a debate. "
            "Below are the final answers from Agent 1 and Agent 2:\n\n"
            f"Agent 1 Final Answer:\n{agent1_final}\n\n"
            f"Agent 2 Final Answer:\n{agent2_final}\n\n"
            "Analyze the responses and determine if the agents have reached consensus. "
            "Consider any subtle contradictions or differences. "
            "Respond with a JSON object with keys 'consensus' (true/false) and 'explanation' (detailed analysis)."
        )
        
        try:
            raw_response = self.generate_response(analysis_prompt)
            
            # Try to extract JSON from the response if it's embedded in text
            import re
            json_match = re.search(r'{[\s\S]*?}', raw_response)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                result = json.loads(raw_response)
                
            consensus = result.get("consensus", False)
            explanation = result.get("explanation", "No explanation provided.")
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON from moderator response: %s", raw_response)
            consensus = False
            explanation = f"The moderator could not provide a structured analysis. Raw response: {raw_response[:500]}..."
        except Exception as e:
            logging.error("Moderator analysis error: %s", e)
            consensus = False
            explanation = f"Failed to parse moderator analysis. The raw response was: {raw_response[:500]}..."
        
        return consensus, explanation

    def generate_response(self, prompt: str, context: List[Dict] = None) -> str:
        """
        Generate a response using the configured provider's API with error handling.
        Returns a JSON string with 'consensus' and 'explanation' for debate moderation.
        """
        if not self.api_key:
            logging.error("No API key provided for Moderator (%s)", self.provider)
            st.error(f"Moderator API key missing for {self.provider}. Please provide it in the sidebar.")
            return json.dumps({
                "consensus": False,
                "explanation": "Error: No API key provided."
            })

        # System instruction for consistent moderation behavior
        system_prompt = (
            f"You are {self.name}, a moderator AI tasked with analyzing debate responses. "
            "Your role is to determine if consensus is reached between two agents. "
            "Respond with a JSON object containing 'consensus' (true/false) and 'explanation' (detailed analysis)."
        )

        # Prepare the messages based on provider
        if self.provider == "OpenAI":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            messages = [{"role": "system", "content": system_prompt}]
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": "gpt-4o",  # Update to latest model as needed
                "messages": messages,
                "temperature": 0.5,  # Lower temperature for analytical tasks
                "response_format": {"type": "json_object"}  # OpenAI-specific JSON mode
            }
            api_url = "https://api.openai.com/v1/chat/completions"

            try:
                response = api_call_with_retry(api_url, headers, payload)
                response_text = response["choices"][0]["message"]["content"]
            except Exception as e:
                logging.error("OpenAI API error for Moderator: %s", e)
                st.error("Moderator failed to analyze with OpenAI.")
                return json.dumps({"consensus": False, "explanation": f"Error: {str(e)}"})

        elif self.provider == "Anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            full_prompt = system_prompt + "\n\n" + prompt + "\n\nEnsure your response is a valid JSON object."
            if context:
                for msg in context:
                    full_prompt += f"\n\n{msg['role'].capitalize()}: {msg['content']}"
            payload = {
                "model": "claude-3-sonnet-20240229",  # Update to latest model
                "max_tokens": 500,
                "temperature": 0.5,
                "messages": [{"role": "user", "content": full_prompt}]
            }
            api_url = "https://api.anthropic.com/v1/messages"

            try:
                response = api_call_with_retry(api_url, headers, payload)
                response_text = response["content"][0]["text"]
            except Exception as e:
                logging.error("Anthropic API error for Moderator: %s", e)
                st.error("Moderator failed to analyze with Anthropic.")
                return json.dumps({"consensus": False, "explanation": f"Error: {str(e)}"})

        elif self.provider == "Gemini":
            try:
                client = genai.Client(api_key=self.api_key)
                full_prompt = system_prompt + "\n\n" + prompt
                if context:
                    for msg in context:
                        full_prompt += f"\n\n{msg['role'].capitalize()}: {msg['content']}"
                response = client.models.generate_content(
                    model="gemini-2.0-pro-exp-02-05",
                    contents=full_prompt,
                    config={"response_mime_type": "application/json"}  # Request JSON response
                )
                response_text = response.text
            except Exception as e:
                logging.error("Gemini API error for Moderator: %s", e)
                st.error("Moderator failed to analyze with Gemini.")
                return json.dumps({"consensus": False, "explanation": f"Error: {str(e)}"})

        else:
            logging.error("Unsupported provider for Moderator: %s", self.provider)
            return json.dumps({"consensus": False, "explanation": f"Error: Unsupported provider {self.provider}"})

        # Ensure the response is valid JSON and update history
        try:
            json.loads(response_text)  # Verify it’s valid JSON
            self.update_history("user", prompt)
            self.update_history("assistant", response_text)
            return response_text
        except json.JSONDecodeError:
            logging.error("Moderator response is not valid JSON: %s", response_text)
            return json.dumps({
                "consensus": False,
                "explanation": "Error: Invalid JSON response from API."
            })


class DebateOrchestrator:
    """Orchestrates the debate between two AI agents."""
    def __init__(self, agent1: AIAgent, agent2: AIAgent, moderator: ModeratorAgent, max_turns: int = 5):
        self.agent1 = agent1
        self.agent2 = agent2
        self.moderator = moderator
        self.max_turns = max_turns
        self.debate_history = []

    def run_debate(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Handle the debate sequence, back-and-forth responses, and consensus checking.
        """
        self.debate_history = []
        
        try:
            # Agent 1 responds to the initial prompt.
            agent1_response = self.agent1.generate_response(initial_prompt)
            self.debate_history.append({"agent": self.agent1.name, "content": agent1_response})
            
            # Agent 2 responds to Agent 1.
            agent2_prompt = (
                f"Initial question: {initial_prompt}\n\n"
                f"{self.agent1.name}'s response: {agent1_response}\n\n"
                f"Analyze {self.agent1.name}'s response and provide your own perspective."
            )
            agent2_response = self.agent2.generate_response(agent2_prompt)
            self.debate_history.append({"agent": self.agent2.name, "content": agent2_response})
            
            turn = 1
            consensus_reached = False
            while turn < self.max_turns and not consensus_reached:
                context_for_agent1 = (
                    f"Previous response from {self.agent2.name}: {agent2_response}\n\n"
                    "Consider this perspective and refine your answer."
                )
                agent1_response = self.agent1.generate_response(context_for_agent1)
                self.debate_history.append({"agent": self.agent1.name, "content": agent1_response})
                
                context_for_agent2 = (
                    f"Previous response from {self.agent1.name}: {agent1_response}\n\n"
                    "Consider this perspective and refine your answer."
                )
                agent2_response = self.agent2.generate_response(context_for_agent2)
                self.debate_history.append({"agent": self.agent2.name, "content": agent2_response})
                
                if turn >= self.max_turns - 1:
                    # Final responses
                    final_prompt_1 = f"Based on the discussion so far, provide your final answer to: {initial_prompt}"
                    final_answer_1 = self.agent1.generate_response(final_prompt_1)
                    self.debate_history.append({"agent": self.agent1.name, "content": final_answer_1, "is_final": True})
                    
                    final_prompt_2 = f"Based on the discussion so far, provide your final answer to: {initial_prompt}"
                    final_answer_2 = self.agent2.generate_response(final_prompt_2)
                    self.debate_history.append({"agent": self.agent2.name, "content": final_answer_2, "is_final": True})
                    
                    consensus_reached, moderator_explanation = self.moderator.check_consensus(final_answer_1, final_answer_2)
                    self.debate_history.append({"agent": self.moderator.name, "content": moderator_explanation, "is_final": True})
                    break
                
                turn += 1
            
            final_result = {
                "initial_prompt": initial_prompt,
                "debate_history": self.debate_history,
                "consensus_reached": consensus_reached,
                "final_answer": self.debate_history[-1]["content"] if consensus_reached else "No consensus reached"
            }
            logging.info("Debate completed. Result: %s", final_result)
            return final_result

        except Exception as e:
            logging.error("Debate orchestration error: %s", e)
            st.error("An unexpected error occurred during the debate. Please try again later.")
            return {"error": "Debate failed due to an internal error."}

# Streamlit UI
def main():
    st.set_page_config(page_title="AITalkers Reasoning System", page_icon="🤖", layout="wide")
    st.title("AI Talkers Reasoning")
    st.subheader("Two AI agents reason until they reach consensus")

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Agent 1")
        agent1_provider = st.selectbox("Provider for Agent 1", ["OpenAI", "Anthropic", "Gemini"], index=0)
        agent1_name = st.text_input("Name for Agent 1", "Logical Analyzer")

        st.subheader("Agent 2")
        agent2_provider = st.selectbox("Provider for Agent 2", ["Anthropic", "OpenAI", "Gemini"], index=0)
        agent2_name = st.text_input("Name for Agent 2", "Critical Thinker")

        st.subheader("Moderator")
        moderator_provider = st.selectbox("Provider for Moderator", ["OpenAI", "Anthropic", "Gemini"], index=0)
        moderator_name = st.text_input("Name for Moderator", "Consensus Detector")

        st.subheader("Debate Settings")
        max_turns = st.slider("Maximum Debate Turns", min_value=1, max_value=10, value=3)

        st.subheader("API Keys (Optional)")
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

    st.header("Enter your question for dual AI reasoning")
    user_prompt = st.text_area("Prompt", "What is the best way to integrate AI into robots (embodied AI)?", height=100)

    if st.button("Start Debate"):
        with st.spinner("Initializing agents..."):
            if agent1_provider == "OpenAI":
                agent1 = OpenAIAgent(agent1_name, openai_api_key, "OpenAI")
            elif agent1_provider == "Anthropic":
                agent1 = AnthropicAgent(agent1_name, anthropic_api_key, "Anthropic")
            else:  # Gemini
                agent1 = GeminiAgent(agent1_name, gemini_api_key, "Gemini")

            if agent2_provider == "OpenAI":
                agent2 = OpenAIAgent(agent2_name, openai_api_key, "OpenAI")
            elif agent2_provider == "Anthropic":
                agent2 = AnthropicAgent(agent2_name, anthropic_api_key, "Anthropic")
            else:  # Gemini
                agent2 = GeminiAgent(agent2_name, gemini_api_key, "Gemini")

            if moderator_provider == "OpenAI":
                moderator = ModeratorAgent(moderator_name, openai_api_key, "OpenAI")
            elif moderator_provider == "Anthropic":
                moderator = ModeratorAgent(moderator_name, anthropic_api_key, "Anthropic")
            else:  # Gemini
                moderator = ModeratorAgent(moderator_name, gemini_api_key, "Gemini")

            orchestrator = DebateOrchestrator(agent1, agent2, moderator, max_turns)

        with st.spinner("Debate in progress..."):
            result = orchestrator.run_debate(user_prompt)

        st.header("Debate Results")
        tab1, tab2 = st.tabs(["Debate", "Final Answer"])
        with tab1:
            st.subheader("Debate History")
            for i, entry in enumerate(result.get("debate_history", [])):
                if entry.get("is_final"):
                    prefix = "📝 FINAL ANSWER: "
                else:
                    prefix = ""
                st.markdown(f"**{entry['agent']}**")
                st.markdown(f'<div style="padding: 10px; border-radius: 5px; background-color: rgba(0, 0, 255, 0.1);">{prefix}{entry["content"]}</div>', unsafe_allow_html=True)
                st.markdown("---")
        with tab2:
            st.subheader("Final Consensus")
            st.success(f"Consensus Reached: {result.get('consensus_reached', False)}")
            st.info(result.get("final_answer", ""))
            st.download_button(
                label="Download Debate as JSON",
                data=json.dumps(result, indent=2),
                file_name="ai_debate_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
