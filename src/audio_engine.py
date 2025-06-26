# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Multi-Threaded Audio Engine
# ==============================================================================

import os
import time
import threading
import tempfile

class AudioEngineManager:
    """
    Manages non-blocking audio alerts via Pygame audio mixer, pyttsx3 offline TTS,
    or gTTS text-to-speech, implementing cooldown timers to prevent stuttering.
    """
    def __init__(self, cooldown_seconds=2.0, enable_tts=False, use_offline_tts=True):
        self.cooldown_seconds = cooldown_seconds
        self.enable_tts = enable_tts
        self.use_offline_tts = use_offline_tts
        self.last_alert_time = 0
        self.last_announced_class = ""
        self._pygame_initialized = False
        self._tts_engine = None

        if self.enable_tts and self.use_offline_tts:
            self._init_pyttsx3()

    def _init_pygame(self):
        if not self._pygame_initialized:
            try:
                import pygame
                pygame.mixer.init()
                self._pygame_initialized = True
            except ImportError:
                print("[WARN] Pygame is not installed. Audio playback disabled.")
            except Exception as e:
                print(f"[WARN] Failed to initialize Pygame mixer: {e}")

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', 165)
        except ImportError:
            print("[WARN] pyttsx3 is not installed. Falling back to gTTS/chime mode.")
            self.use_offline_tts = False
        except Exception as e:
            print(f"[WARN] Could not initialize pyttsx3 engine: {e}")
            self.use_offline_tts = False

    def trigger_alert(self, class_name, sound_path=None):
        """
        Triggers an audio notification (chime audio file or spoken voice alert).
        Executes on a background daemon thread so video processing never freezes.
        """
        current_time = time.time()
        # Cooldown check
        if (current_time - self.last_alert_time < self.cooldown_seconds) and (class_name == self.last_announced_class):
            return

        self.last_alert_time = current_time
        self.last_announced_class = class_name

        def _worker():
            if self.enable_tts:
                self._speak(class_name)
            elif sound_path and os.path.exists(sound_path):
                self._play_file(sound_path)
            else:
                self._speak(class_name)

        threading.Thread(target=_worker, daemon=True).start()

    def _play_file(self, sound_path):
        self._init_pygame()
        if self._pygame_initialized:
            try:
                import pygame
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"[ERROR] Audio file playback failed: {e}")

    def _speak(self, text_to_speak):
        msg = f"{text_to_speak} detected"
        if self.use_offline_tts and self._tts_engine is not None:
            try:
                self._tts_engine.say(msg)
                self._tts_engine.runAndWait()
                return
            except Exception as e:
                print(f"[WARN] Offline TTS failed: {e}")

        # Fallback to gTTS online text-to-speech
        try:
            from gtts import gTTS
            tts = gTTS(text=msg, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
            tts.save(temp_path)

            self._play_file(temp_path)
            time.sleep(1.0)

            if os.path.exists(temp_path):
                os.remove(temp_path)
        except ImportError:
            print(f"[ALERT] *** {msg.upper()} *** (Audio packages not installed)")
        except Exception as e:
            print(f"[ERROR] Text-to-speech failed: {e}")
