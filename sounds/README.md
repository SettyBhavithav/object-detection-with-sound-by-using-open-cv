# Audio Asset Directory

This folder contains audio notification alert files played by the multi-threaded audio engine upon detecting target objects:

- **`alert.wav`**: Standard proximity alert chime sound.
- **`output.mp3`**: Generated text-to-speech audio sample.

Audio playback is managed asynchronously using `pygame.mixer` or `pyttsx3`/`gTTS`.
