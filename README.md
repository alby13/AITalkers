<center><img src="https://github.com/alby13/AITalkers/blob/main/ai-talkers-logo.jpg"></center>


# AITalkers Dual Agent Reasoning Framework
LLM Framework for Two AI Agents Debate/Reasoning to reach Consensus Answers

AITalkers enables two AI agents that are configurable to use providers like OpenAI, Anthropic, or Gemini to engage in a structured reasoning debate. A moderator agent analyzes the final answers to review them and determine whether consensus has been reached.


<br><br>
## This is an early version release that still may have bugs and requires additional testing and refinement.

## Features

- **Dual AI Debate:** Two agents debate on a given prompt, iteratively refining their responses.
- **Moderator Analysis:** An AI moderator checks for consensus between the agents using JSON-based analysis.
- **Customizable Providers:** Select among OpenAI, Anthropic, and Gemini for both agents and the moderator.
- **Interactive UI:** A user-friendly Streamlit interface for configuration, debate execution, and viewing results.


<br><br>
## Conversational AI Agents:

- Independent Agent Classes:
Each AI agent (OpenAIAgent, AnthropicAgent, ModeratorAgent) is defined as an independent class with its own state, memory (conversation history), and behavior (response generation). This encapsulation is a core trait of agentic systems, where each “agent” can operate based on its internal logic.

- Autonomous Response Generation:
The agents are designed to generate responses based on recent context. They update their own state (by maintaining full conversation history) and handle context truncation autonomously, which is a hallmark of agentic behavior.

- Centralized Orchestration with Decentralized Autonomy:
The agents operate independently, their interactions are coordinated by a central Debate Orchestrator. This orchestrator manages turn-taking and the overall debate flow. <br>This framework's design intentionally imposes structure to maintain coherent debate dynamics, which is common in many practical multi-agent systems.

#### Agentic Framework:
This framework is agentic in that it leverages independent, stateful agents that act according to their own internal logic. At the same time, it uses a central orchestrator to coordinate interactions, making it a hybrid design—agentic at the individual level but structured at the system level. This balance is desirable in this application for debates, where both autonomy and coordination are key.


<br><br>
## Installation
Requires Python 3.10+

Before using you will have to set up the exact AI model name of each type that you would like to use in the code (OpenAI, Claude, Gemini).


### Prerequisites

- Python 3.10 or later
- Pip (Python package manager)
  
- (Optional) Virtual environment tool (such as `venv` or `conda`)

<br><br>
### Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/alby13/AITalkers.git
cd AITalkers
```

<br><br>
### Installing on Windows

Open Command Prompt or PowerShell.

(Optional) Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install Dependencies:

If the requirements.txt file is present in your directory, run:

```
pip install -r requirements.txt
```


### Installing on Linux

Open a Terminal.

(Optional) Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

### Configure API Keys:
Create a .env file in the repository root and add your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

Install Dependencies:

If the requirements.txt file is present in your directory, run:
```
pip install -r requirements.txt
```

Otherwise, install the required packages manually:

```
pip install streamlit requests python-dotenv
```

## Usage
After installation and configuration, you can start the AITalkers application using Streamlit.

```
streamlit run AITalkers.py
```

<br><br>
## Future Improvements

- Early Consensus Checking
- History/Records/Logging
- Easier way for user to specify AI Models

- Streamlit UI Enhancements
Add a progress bar for the debate:

<code>progress_bar = st.progress(0)
for turn in range(max_turns):
    progress_bar.progress((turn + 1) / max_turns)
    # Debate logic...</code>
    
<br><br>
## Contributing & Evaluations
Contributions and Evaluations are welcome! If you have any ideas for improvements or encounter any issues, please open an issue or submit a pull request.

<br><br>
## MIT License

Copyright (c) 2025 alby13

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
