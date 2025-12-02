import pyaudio
import pygame
import edge_tts
import wave
from mail_send import send_email
import time
import pickle
import os
import base64
import re
import html
from typing import List, Dict, Optional
import json
import asyncio
import io
import dotenv
dotenv.load_dotenv()
import speech_recognition as sr
import sounddevice as sd
# Audio processing
# from faster_whisper import WhisperModel
import numpy as np
import struct
import pvporcupine
# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LLM API
import requests

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = "voice.wav"

# LLM Configuration
API_KEY = "1c4b22fb-2066-4cb7-b7cf-bff96ff74331"
API_URL = "https://api.sambanova.ai/v1/chat/completions"
 
# Voice Options
VOICE_MAP = {
    "1": "en-US-JennyNeural",
    "2": "en-GB-SoniaNeural", 
    "3": "fr-FR-HenriNeural",
    "4": "en-US-AriaNeural",
    "5": "hi-IN-SwaraNeural",
    "6": "hi-IN-MadhurNeural",
}
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": """**You are a highly personalized AI assistant trained specifically for a single user.**
> You have access to the user's recent emails, preferences, and interaction history.
> Your job is to respond conversationally, clearly, and helpfully.
> Always try to **understand the user's intent**, fetch or process the needed info, and **suggest meaningful actions based on context**.
>
> When replying:
>
> * Be friendly but professional.
> * **Proactively suggest things** the user might want to do next.
> * If discussing emails, summarize and **suggest replies, meetings, tasks, or follow-ups.**
> * If asked about news, weather, tools, or recommendations, **personalize your answer based on their interests**.
> * Use natural, human-like tone — like a chill but smart assistant who gets things done.
>
> Example:
>
> * **User**: Tell me the latest emails
> * **Assistant**: You’ve got 3 new emails — one from your professor about tomorrow’s lab, another from Amazon about a delivery, and one from Skillshare on a new course. Want me to draft a reply to your professor or remind you to attend the lab?"""
}
chat_history = [SYSTEM_MESSAGE]
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY") or "Kr71SaOvrasJcijvz66Al9TCnNixLowzmVbhWqlfWEXrjuMVDPaShg=="
WAKE_WORD = "jarvis"
class GmailVoiceBot:
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.pickle'):
        """Initialize Gmail Voice Bot"""
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.sambanova_api_key = API_KEY
        self.sambanova_url = API_URL
        self.is_authenticated = False        
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()
        try:
            # Handle built-in keywords or custom paths
            if WAKE_WORD.endswith('.ppn'):
                keyword_paths = [WAKE_WORD]
                keywords = None
            else:
                keyword_paths = None
                keywords = [WAKE_WORD]

            self.porcupine = pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
                keyword_paths=keyword_paths,
                keywords=keywords
            )
            self.audio_stream_wake_word = pyaudio.PyAudio().open(
                rate=self.porcupine.sample_rate,
                channels=1, # Mono
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            print(f"✅ Wake word engine initialized for '{WAKE_WORD}'")
        except Exception as e:
            print(f"❌ Error initializing wake word engine: {e}")
            self.porcupine = None
            self.audio_stream_wake_word = None
        
    def authenticate_gmail(self) -> bool:
        """Authenticate with Gmail API"""
        if self.is_authenticated:
            return True
            
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_file):
                    print(f"❌ Gmail credentials file '{self.credentials_file}' not found!")
                    return False
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.scopes)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"❌ Gmail authentication error: {e}")
                    return False
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            self.service = build('gmail', 'v1', credentials=creds)
            self.is_authenticated = True
            print("✅ Gmail API authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Error building Gmail service: {e}")
            return False
    
    def _extract_message_body(self, payload: Dict) -> str:
        """Extract text body from message payload"""
        body = ""
        
        if 'parts' in payload:
            # Multipart message
            for part in payload['parts']:
                body += self._extract_message_body(part)
        else:
            # Single part message
            if payload.get('mimeType') == 'text/plain':
                data = payload.get('body', {}).get('data', '')
                if data:
                    try:
                        decoded = base64.urlsafe_b64decode(data).decode('utf-8')
                        body += decoded
                    except Exception:
                        pass
            
            elif payload.get('mimeType') == 'text/html':
                data = payload.get('body', {}).get('data', '')
                if data:
                    try:
                        decoded = base64.urlsafe_b64decode(data).decode('utf-8')
                        clean_text = re.sub('<.*?>', '', decoded)
                        clean_text = html.unescape(clean_text)
                        body += clean_text
                    except Exception:
                        pass
        
        return body.strip()
    
    def get_gmail_profile(self) -> Dict:
        """Get user's Gmail profile"""
        if not self.authenticate_gmail():
            return {"error": "Authentication failed"}
            
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email': profile.get('emailAddress'),
                'messages_total': profile.get('messagesTotal'),
                'threads_total': profile.get('threadsTotal')
            }
        except HttpError as error:
            return {"error": f"Error getting profile: {error}"}
    
    def get_latest_emails(self, count: int = 5, unread_only: bool = False) -> List[Dict]:
        """Get latest emails"""
        if not self.authenticate_gmail():
            return [{"error": "Authentication failed"}]
            
        try:
            query = 'is:unread' if unread_only else ''
            result = self.service.users().messages().list(
                userId='me', 
                q=query, 
                maxResults=count
            ).execute()
            
            messages = result.get('messages', [])
            detailed_messages = []
            
            for msg in messages:
                try:
                    message = self.service.users().messages().get(
                        userId='me', 
                        id=msg['id'], 
                        format='full'
                    ).execute()
                    
                    payload = message['payload']
                    headers = payload.get('headers', [])
                    
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                    date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                    
                    body = self._extract_message_body(payload)
                    
                    detailed_messages.append({
                        'id': msg['id'],
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'snippet': message.get('snippet', ''),
                        'body': body[:500] + ('...' if len(body) > 500 else ''),
                        'labels': message.get('labelIds', [])
                    })
                except Exception:
                    continue
                    
            return detailed_messages
            
        except HttpError as error:
            return [{"error": f"Error listing messages: {error}"}]
    
    def search_emails(self, query: str, count: int = 5) -> List[Dict]:
        """Search emails with query"""
        if not self.authenticate_gmail():
            return [{"error": "Authentication failed"}]
            
        try:
            result = self.service.users().messages().list(
                userId='me', 
                q=query, 
                maxResults=count
            ).execute()
            
            messages = result.get('messages', [])
            detailed_messages = []
            
            for msg in messages:
                try:
                    message = self.service.users().messages().get(
                        userId='me', 
                        id=msg['id'], 
                        format='full'
                    ).execute()
                    
                    payload = message['payload']
                    headers = payload.get('headers', [])
                    
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                    date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                    
                    detailed_messages.append({
                        'id': msg['id'],
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'snippet': message.get('snippet', '')
                    })
                except Exception:
                    continue
                    
            return detailed_messages
            
        except HttpError as error:
            return [{"error": f"Error searching emails: {error}"}]
    
    def summarize_email(self, email_content: Dict) -> str:
        """Summarize email using AI"""
        try:
            content_text = f"""
            Subject: {email_content.get('subject', 'No Subject')}
            From: {email_content.get('sender', 'Unknown')}
            Date: {email_content.get('date', 'Unknown')}
            
            Email Content:
            {email_content.get('body', '')[:1000]}
            """
            
            headers = {
                "Authorization": f"Bearer {self.sambanova_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "Meta-Llama-3.3-70B-Instruct",
                "messages": [
                    {
                        "role": "system", 
                        "content": '''You are a highly personalized AI assistant trained specifically for a single user.
                                > Your job is to respond conversationally, clearly, and helpfully
                                > When replying:
                                > * Be friendly.
                                > * If asked about news, weather, tools, or recommendations.
                                > * Use natural, human-like tone — like a chill but smart assistant who gets things done.
                                > * complete your response in only 150 words
                            '''
                    },
                    {
                        "role": "user", 
                        "content": f"Please summarize this email:\n\n{content_text}"
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }
            
            response = requests.post(self.sambanova_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error with AI API: {response.status_code}"
                
        except Exception as e:
            return f"Error summarizing email: {e}"
    
    def process_gmail_command(self, command: str) -> str:
        """Process Gmail-related voice commands"""
        command = command.lower().strip()

        try:
            # Check my email / Get latest emails
            if any(phrase in command for phrase in ["check my email", "latest email", "recent email", "new email", "get latest", "show latest", "latest emails", "give me latest", "can you give me"]):
                print("📧 Fetching latest emails...")
                emails = self.get_latest_emails(count=3)
                if not emails:
                    return "You have no emails in your inbox."
                
                if isinstance(emails, list) and len(emails) > 0 and "error" in emails[0]:
                    return f"Sorry, I couldn't access your emails. {emails[0]['error']}"
                
                response = f"You have {len(emails)} recent emails. Here they are: "
                for i, email in enumerate(emails, 1):
                    sender_name = email['sender'].split('<')[0].strip() if '<' in email['sender'] else email['sender']
                    response += f"Email {i}: From {sender_name}, Subject: {email['subject']}. "
                
                return response
            
            # Check unread emails
            elif any(phrase in command for phrase in ["unread email", "new message", "check unread", "unread messages"]):
                print("📬 Checking unread emails...")
                emails = self.get_latest_emails(count=5, unread_only=True)
                if not emails:
                    return "You have no unread emails. Great job staying on top of your inbox!"
                
                if isinstance(emails, list) and len(emails) > 0 and "error" in emails[0]:
                    return f"Sorry, I couldn't check your unread emails. {emails[0]['error']}"
                
                response = f"You have {len(emails)} unread emails. "
                for i, email in enumerate(emails, 1):
                    sender_name = email['sender'].split('<')[0].strip() if '<' in email['sender'] else email['sender']
                    response += f"Unread email {i}: From {sender_name}, Subject: {email['subject']}. "
                
                return response
            
            # Search emails
            elif any(phrase in command for phrase in ["search email", "find email", "look for email", "search for"]):
                print("🔍 Searching emails...")
                # Extract search terms (simple implementation)
                if "from" in command:
                    # Extract sender
                    search_query = "from:" + command.split("from")[-1].strip()
                elif "subject" in command:
                    # Extract subject
                    search_query = "subject:" + command.split("subject")[-1].strip()
                else:
                    # General search
                    search_terms = command.replace("search email", "").replace("find email", "").replace("look for email", "").replace("search for", "").strip()
                    search_query = search_terms
                
                emails = self.search_emails(search_query, count=3)
                if not emails:
                    return f"No emails found for your search: {search_query}"
                
                if isinstance(emails, list) and len(emails) > 0 and "error" in emails[0]:
                    return f"Sorry, I couldn't search your emails. {emails[0]['error']}"
                
                response = f"Found {len(emails)} emails matching your search. "
                for i, email in enumerate(emails, 1):
                    sender_name = email['sender'].split('<')[0].strip() if '<' in email['sender'] else email['sender']
                    response += f"Result {i}: From {sender_name}, Subject: {email['subject']}. "
                
                return response
            
            # Summarize latest email
            elif any(phrase in command for phrase in ["summarize", "summary of", "what does it say", "tell me about"]):
                print("🤖 Generating email summary...")
                emails = self.get_latest_emails(count=1)
                if not emails:
                    return "You have no emails to summarize."
                
                if isinstance(emails, list) and len(emails) > 0 and "error" in emails[0]:
                    return f"Sorry, I couldn't access your emails for summarization. {emails[0]['error']}"
                
                latest_email = emails[0]
                summary = self.summarize_email(latest_email)
                sender_name = latest_email['sender'].split('<')[0].strip() if '<' in latest_email['sender'] else latest_email['sender']
                return f"Here's a summary of your latest email from {sender_name}: {summary}"
            
            # Get email count/profile
            elif any(phrase in command for phrase in ["how many email", "email count", "my profile", "total emails"]):
                print("📊 Getting Gmail profile...")
                profile = self.get_gmail_profile()
                if "error" in profile:
                    return f"Sorry, I couldn't get your Gmail profile. {profile['error']}"
                
                return f"Your Gmail account {profile['email']} has {profile['messages_total']} total messages and {profile['threads_total']} conversation threads."
            
            else:
                return "I can help you check emails, search emails, get unread emails, summarize emails, or get your email count. What would you like to do with your Gmail?"
                
        except Exception as e:
            return f"Sorry, I encountered an error processing your Gmail request: {str(e)}"
    
    # def record_audio(self):
    #     """Record audio from microphone"""
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=FORMAT,
    #                     channels=CHANNELS,
    #                     rate=RATE,
    #                     input=True,
    #                     frames_per_buffer=CHUNK)
        
    #     print("\n🎤 Recording... Press Enter to stop")
    #     frames = []
        
    #     try:
    #         while True:
    #             data = stream.read(CHUNK, exception_on_overflow=False)
    #             frames.append(data)
                
    #             if sys.platform != 'win32':
    #                 if select.select([sys.stdin], [], [], 0)[0]:
    #                     if sys.stdin.readline():
    #                         break
    #             else:
    #                 import msvcrt
    #                 if msvcrt.kbhit() and msvcrt.getch() == b'\r':
    #                     break
                        
    #     except KeyboardInterrupt:
    #         pass
        
    #     stream.stop_stream()
    #     stream.close()
    #     p.terminate()
        
    #     if frames:
    #         with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    #             wf.setnchannels(CHANNELS)
    #             wf.setsampwidth(p.get_sample_size(FORMAT))
    #             wf.setframerate(RATE)
    #             wf.writeframes(b''.join(frames))
    #         return True
    #     return False

    def calculate_rms(self, data):
        # Convert audio bytes to numbers and calculate RMS
        count = len(data) // 2  # 2 bytes per sample
        format = "%dh" % count
        shorts = struct.unpack(format, data)
        
        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0 / 32768)
            sum_squares += n * n
        rms = (sum_squares / count) ** 0.5 * 1000
        return rms

    def record_audio_with_vad(self):
        """Record audio with voice activity detection - stops after 2 seconds of silence"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            p.terminate()
            return False
        
        print("\n🎤 Listening... Speak now (recording will stop automatically)")
        frames = []
        
        # VAD parameters
        SILENCE_THRESHOLD = 10  # Adjust based on your environment
        SILENCE_DURATION = 2.0   # Stop after 2 seconds of silence
        MIN_RECORDING_TIME = 0.5 # Minimum recording time
        MAX_RECORDING_TIME = 30  # Maximum recording time (safety)
        
        # State tracking
        recording_start = time.time()
        silence_start = None
        speech_detected = False
        last_rms = 0
        
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate RMS for this chunk
                rms = self.calculate_rms(data)
                current_time = time.time()
                recording_time = current_time - recording_start
                
                # Check if we're currently hearing speech
                if rms > SILENCE_THRESHOLD:
                    speech_detected = True
                    silence_start = None  # Reset silence timer
                    print(f"🗣️ Speaking... (Volume: {int(rms):4d})    ", end='\r')
                    last_rms = rms
                else:
                    # We're in silence
                    if speech_detected and silence_start is None:
                        # Just started being silent after speech
                        silence_start = current_time
                        print(f"⏸️ Silence detected... ({int(rms):4d})     ", end='\r')
                    
                    # Check if we should stop recording
                    if (silence_start and 
                        current_time - silence_start >= SILENCE_DURATION and 
                        recording_time >= MIN_RECORDING_TIME and
                        speech_detected):
                        print(f"\n✅ Recording stopped - {SILENCE_DURATION}s of silence detected")
                        break
                
                # Safety timeout
                if recording_time > MAX_RECORDING_TIME:
                    print(f"\n⏰ Maximum recording time ({MAX_RECORDING_TIME}s) reached")
                    break
                
                # Show some feedback for very quiet environments
                if not speech_detected and recording_time > 3:
                    print(f"🔇 Waiting for speech... (Current: {int(rms):4d}, Threshold: {SILENCE_THRESHOLD})", end='\r')
                            
        except KeyboardInterrupt:
            print(f"\n⏹️ Recording stopped by user")
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        # Save audio if we recorded something
        if frames and speech_detected:
            try:
                with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                print(f"💾 Audio saved ({len(frames)} chunks, {recording_time:.1f}s)")
                return True
            except Exception as e:
                print(f"❌ Error saving audio: {e}")
                return False
        elif not speech_detected:
            print("❌ No speech detected - please speak louder or check microphone")
            return False
        else:
            print("❌ No audio recorded")
            return False

    def transcribe_audio(self, audio_file):
        """Transcribe audio to text"""
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"\nYou said: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                return None
    
    # def transcribe_audio(self):
    #     """Transcribe recorded audio to text"""
    #     segments, _ = self.whisper_model.transcribe(WAVE_OUTPUT_FILENAME)
    #     return " ".join(segment.text for segment in segments)
    
    def get_llm_response(self, prompt):
        """Get response from LLM for general queries"""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Meta-Llama-3.3-70B-Instruct",
            "messages": chat_history,
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 150
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        assistant_message = response_data["choices"][0]["message"]
        chat_history.append(assistant_message)
        return json.loads(response.content)['choices'][0]['message']['content']
    
    async def speak(self, text, voice):
        """Convert text to speech and play it"""
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            
            sound = pygame.mixer.Sound(io.BytesIO(audio_data))
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"Speech error: {e}")
    
    def select_voice(self):
        """Select voice for text-to-speech"""
        print("\n🎵 Select a voice:")
        for key, value in VOICE_MAP.items():
            print(f"{key}: {value.split('-')[-1]}")
        while True:
            choice = input("Enter choice (1-4): ")
            if choice in VOICE_MAP:
                return VOICE_MAP[choice]
            print("Invalid choice, please try again.")
    
    async def main_loop(self):
        """Main voice bot loop with voice activity detection"""
        print("🚀 Gmail Voice Assistant")
        print("=" * 50)
        
        # Authenticate Gmail
        if not self.authenticate_gmail():
            print("⚠️ Gmail authentication failed. Some features may not work.")
        
        # Select voice
        voice = self.select_voice()
        
        # Welcome message
        welcome_msg = "Hello! I'm your Gmail voice assistant. How can I help you today?"
        print(f"\n🤖 Assistant: {welcome_msg}")
        await self.speak(welcome_msg, voice)
        WAKE_WORD = "jarvis"
        # Main interaction loop
        while True:
            try:
                print(f"\n{'='*50}")
                print("🎯 Ready for your command...")
                keyword_detected = False
                while not keyword_detected:
                    pcm = self.audio_stream_wake_word.read(self.porcupine.frame_length)
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    keyword_index = self.porcupine.process(pcm)
                    if keyword_index >= 0:
                        print(f"🎯 Wake word '{WAKE_WORD}' detected!")
                        keyword_detected = True
                        # Brief pause after wake word detection
                        time.sleep(0.2)
                
                # Record user input with voice activity detection
                recording_success = self.record_audio_with_vad()
                
                if not recording_success:
                    retry_msg = "I didn't catch that. Could you please try again?"
                    print(f"🤖 Assistant: {retry_msg}")
                    await self.speak(retry_msg, voice)
                    continue
                
                # Transcribe audio
                print("🔄 Processing your speech...")
                user_input = self.transcribe_audio(WAVE_OUTPUT_FILENAME)
                
                if not user_input or user_input.strip() == "":
                    no_input_msg = "I couldn't understand what you said. Please try again."
                    print(f"🤖 Assistant: {no_input_msg}")
                    await self.speak(no_input_msg, voice)
                    continue
                
                print(f"\n👤 You said: {user_input}")
                chat_history.append({"role": "user", "content": user_input})
                
                # Check for exit commands
                exit_commands = ["exit", "quit", "stop", "goodbye", "bye", "end"]
                if any(cmd in user_input.lower().strip() for cmd in exit_commands):
                    goodbye_msg = "Goodbye! Have a great day!"
                    print(f"🤖 Assistant: {goodbye_msg}")
                    await self.speak(goodbye_msg, voice)
                    break
                
                # Process the command
                print("🧠 Thinking...")
                
                # Categorize the command
                gmail_keywords = ["email", "gmail", "inbox", "message", "unread", "search", 
                                "summarize", "count", "latest", "recent", "new", "mail"]
                mail_send_keywords = ["send", "compose", "draft", "reply", "forward", "write"]
                
                is_mail_send_command = any(keyword in user_input.lower() for keyword in mail_send_keywords)
                is_gmail_command = any(keyword in user_input.lower() for keyword in gmail_keywords)
                
                if is_mail_send_command:
                    # Handle email sending with voice input
                    response = await self.handle_email_sending(voice)
                    
                elif is_gmail_command:
                    # Handle Gmail-related commands
                    response = self.process_gmail_command(user_input)

                else:
                    # Handle general queries
                    response = self.get_llm_response(user_input)
                
                print(f"\n🤖 Assistant: {response}")
                chat_history.append({"role": "assistant", "content": response})
                
                # Speak the response
                await self.speak(response, voice)
                
            except KeyboardInterrupt:
                print("\n\n👋 Voice assistant stopped by user")
                goodbye_msg = "Session ended. Goodbye!"
                await self.speak(goodbye_msg, voice)
                break
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                print(f"❌ Error: {error_msg}")
                await self.speak("I'm sorry, something went wrong. Let's try again.", voice)

    async def handle_email_sending(self, voice):
        """Handle email sending with voice input"""
        try:
            # Get sender email via voice
            await self.speak("Please tell me the sender's email address", voice)
            if not self.record_audio_with_vad():
                return "I couldn't get the sender email. Please try again."
            sender_email = self.transcribe_audio(WAVE_OUTPUT_FILENAME).replace(" ", "").replace("attherate", "@").lower()
            if "@" not in sender_email:
                print("Invalid sender email format. Please try again.")
                self.handle_email_sending(voice)
            print(f"📧 Sender: {sender_email}")
            
            # Get receiver email via voice
            await self.speak("Please tell me the receiver's email address", voice)
            if not self.record_audio_with_vad():
                return "I couldn't get the receiver email. Please try again."
            receiver_email = self.transcribe_audio(WAVE_OUTPUT_FILENAME).replace(" ", "").replace("attherate", "@").lower()
            print(f"📧 Receiver: {receiver_email}")
            
            # Get subject via voice
            await self.speak("What's the subject of the email?", voice)
            if not self.record_audio_with_vad():
                return "I couldn't get the subject. Please try again."
            subject = self.transcribe_audio(WAVE_OUTPUT_FILENAME)
            print(f"📝 Subject: {subject}")
            
            # Get message via voice
            await self.speak("Please tell me the message content", voice)
            if not self.record_audio_with_vad():
                return "I couldn't get the message content. Please try again."
            message = self.transcribe_audio(WAVE_OUTPUT_FILENAME)
            print(f"💬 Message: {message}")
            
            # Confirm before sending
            confirm_msg = f"I'll send an email from {sender_email} to {receiver_email} with subject '{subject}'. Should I send it now?"
            await self.speak(confirm_msg, voice)
            
            if not self.record_audio_with_vad():
                return "I didn't get your confirmation. Email not sent."
                
            confirmation = self.transcribe_audio(WAVE_OUTPUT_FILENAME).lower().replace(" ", "")
            
            if any(word in confirmation for word in ["yes", "send", "okay", "sure", "confirm"]):
                # Send the email
                send_email(sender_email, receiver_email, subject, message)
                return "Email has been sent successfully!"
            else:
                return "Email sending cancelled."
                
        except Exception as e:
            return f"Error during email composition: {str(e)}"

from mail_bot import GmailVoiceBot
import asyncio
import pygame
def main():
    """Main function"""
    bot = GmailVoiceBot()
    
    try:
        asyncio.run(bot.main_loop())
    except KeyboardInterrupt:
        print("\n🛑 Gmail Voice Assistant stopped")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()