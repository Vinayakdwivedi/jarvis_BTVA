BTVA is an AI-powered voice assistant designed to perform intelligent, real-time interactions by integrating Speech-to-Text (STT), Large Language Models (LLM), Tool Calling, and Text-to-Speech (TTS) technologies into a single, seamless pipeline.


Developed during an internship at BitsFlow Technologies Pvt. Ltd., this project aims to create a hybrid, modular AI assistant adaptable for automation and interactive assistance.


🚀 Features

Real-time Voice Interaction: Converts spoken input to text, processes intent, and responds with natural speech.



Intelligent LLM Integration: Leverages SambaNova’s Cloud API (running Llama-3B/8B) for fast inference and context-aware responses.




Tool & Function Calling: The LLM can dynamically decide whether to answer directly or invoke external tools (e.g., fetching emails, checking weather).



Email Automation: Core functionality includes sending emails, retrieving unread messages, generating summaries, and searching through emails.


Browser Control: Integration with Model Context Protocol (MCP) and Microsoft Playwright allows for automated browser control.




High-Quality TTS: Uses Microsoft Edge TTS for lifelike, multilingual speech output with pitch and speed tuning.


Chainlit Web UI: A user-friendly web interface featuring voice selection, visual feedback, and interactive email cards.


🏗️ System Architecture
The system operates on a modular pipeline :

Input: User provides audio input via microphone or file upload.


Speech-to-Text (STT): Audio is transcribed using Google Speech Recognition (optimized for speed) or faster_whisper (optimized for local accuracy).


LLM Processing: The transcript is sent to the SambaNova LLM. The model interprets intent and context.


Tool Execution (Optional): If a task is detected (e.g., "Send an email"), the system executes the specific tool via structured function calls.


Text-to-Speech (TTS): The final text response is synthesized into audio using Edge TTS and played back to the user.

🛠️ Tech Stack
Language: Python


LLM Provider: SambaNova Cloud API (Llama-3.1-8B-Instruct).


Speech-to-Text:


SpeechRecognition (Google Web Speech API).


faster_whisper (Initial implementation).


Text-to-Speech: edge-tts (Microsoft Edge Neural TTS).


UI Framework: Chainlit.


Automation/Tools: Model Context Protocol (MCP), Microsoft Playwright.


⚙️ Configuration & Code Snippets
LLM Request Structure
The project uses the SambaNova API for low-latency responses. Below is the request structure used in the project :
<img width="876" height="444" alt="image" src="https://github.com/user-attachments/assets/26e42b78-1694-420f-a8e9-cadb3687a661" />

Speech Recognition
The system utilizes Google's free speech recognition for optimal speed:
<img width="883" height="167" alt="image" src="https://github.com/user-attachments/assets/e8ca4d68-1af4-4d44-99fa-f36ea038b2b0" />

🖥️ Usage
The application enables both voice-first and text-based interactions.

Voice Commands
You can interact with the bot using natural language. Tested commands include:


"latest email Le jaldi mere ko" (Fetch my latest email quickly).


"unread emails jio jaldi se" (Check unread emails).


"yaar mere ko ek badhiya Sa joke Bata Ki Main Hans ke Gir jaaun" (Tell me a joke so funny I fall over).

Web Interface (Chainlit)

Voice Selection: Choose different TTS voices via a dropdown.


Visual Cards: Emails are displayed as interactive cards rather than raw text.


Action Buttons: Quick actions to reply to or delete emails.

🔮 Future Enhancements

Multigenetic Framework: Modularizing functionality to add calendar integration and document generation easily.


Speech Interruption: allowing the bot to pause or adapt when the user speaks mid-sentence.


Advanced Noise Filtering: Improving transcription accuracy in noisy environments.

