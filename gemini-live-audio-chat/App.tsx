
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality } from '@google/genai';
import { ConversationStatus, TranscriptEntry } from './types';
import { encode, decode, decodeAudioData, createBlob } from './utils/audioUtils';
import { MicrophoneIcon, StopIcon, LoadingSpinnerIcon, UserIcon, GeminiIcon } from './components/Icons';

const App: React.FC = () => {
  const [status, setStatus] = useState<ConversationStatus>(ConversationStatus.IDLE);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  
  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');
  const nextStartTimeRef = useRef(0);
  const playingSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  const stopConversation = useCallback(async () => {
    setStatus(ConversationStatus.PROCESSING);
    try {
      if (sessionPromiseRef.current) {
        const session = await sessionPromiseRef.current;
        session.close();
        sessionPromiseRef.current = null;
      }
    } catch (e) {
      console.error("Error closing session:", e);
    }
    
    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach(track => track.stop());
      microphoneStreamRef.current = null;
    }

    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }

    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
      await inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
      for (const source of playingSourcesRef.current.values()) {
          source.stop();
      }
      playingSourcesRef.current.clear();
      await outputAudioContextRef.current.close();
      outputAudioContextRef.current = null;
    }

    currentInputTranscriptionRef.current = '';
    currentOutputTranscriptionRef.current = '';
    nextStartTimeRef.current = 0;
    
    setStatus(ConversationStatus.IDLE);
  }, []);

  const startConversation = useCallback(async () => {
    if (status !== ConversationStatus.IDLE) return;
    
    setStatus(ConversationStatus.PROCESSING);
    setError(null);
    setTranscript([]);

    try {
      if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set.");
      }
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      microphoneStreamRef.current = stream;

      // Fix for webkitAudioContext not being available on the window type for cross-browser compatibility.
      const CrossBrowserAudioContext = window.AudioContext || (window as any).webkitAudioContext;
      inputAudioContextRef.current = new CrossBrowserAudioContext({ sampleRate: 16000 });
      outputAudioContextRef.current = new CrossBrowserAudioContext({ sampleRate: 24000 });
      
      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: 'You are a helpful and friendly AI assistant. Keep your responses concise and conversational.',
        },
        callbacks: {
          onopen: () => {
            if (!inputAudioContextRef.current) return;
            const source = inputAudioContextRef.current.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              if (sessionPromiseRef.current) {
                sessionPromiseRef.current.then((session) => {
                  session.sendRealtimeInput({ media: pcmBlob });
                });
              }
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current.destination);
            setStatus(ConversationStatus.LISTENING);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTranscriptionRef.current += text;
            } else if (message.serverContent?.inputTranscription) {
              const text = message.serverContent.inputTranscription.text;
              currentInputTranscriptionRef.current += text;
            }

            if (message.serverContent?.turnComplete) {
              const fullInput = currentInputTranscriptionRef.current.trim();
              const fullOutput = currentOutputTranscriptionRef.current.trim();

              if (fullInput) {
                 setTranscript(prev => [...prev, { speaker: 'user', text: fullInput }]);
              }
              if (fullOutput) {
                 setTranscript(prev => [...prev, { speaker: 'gemini', text: fullOutput }]);
              }

              currentInputTranscriptionRef.current = '';
              currentOutputTranscriptionRef.current = '';
            }

            const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audioData && outputAudioContextRef.current) {
              const audioContext = outputAudioContextRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, audioContext.currentTime);
              
              const audioBuffer = await decodeAudioData(
                decode(audioData),
                audioContext,
                24000,
                1
              );

              const source = audioContext.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(audioContext.destination);
              
              source.addEventListener('ended', () => {
                playingSourcesRef.current.delete(source);
              });
              
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              playingSourcesRef.current.add(source);
            }

            const interrupted = message.serverContent?.interrupted;
            if (interrupted && outputAudioContextRef.current) {
              for (const source of playingSourcesRef.current.values()) {
                source.stop();
              }
              playingSourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }
          },
          onerror: (e: ErrorEvent) => {
            console.error('API Error:', e);
            setError(`Connection error: ${e.message}`);
            stopConversation();
          },
          onclose: (e: CloseEvent) => {
            console.log('Connection closed.');
          },
        },
      });

    } catch (err: any) {
      console.error("Failed to start conversation:", err);
      setError(err.message || 'An unexpected error occurred.');
      await stopConversation();
    }
  }, [status, stopConversation]);

  const handleButtonClick = () => {
    if (status === ConversationStatus.IDLE) {
      startConversation();
    } else {
      stopConversation();
    }
  };

  const getStatusText = () => {
    switch (status) {
      case ConversationStatus.IDLE:
        return 'Tap to speak';
      case ConversationStatus.PROCESSING:
        return 'Processing...';
      case ConversationStatus.LISTENING:
        return 'Listening...';
      default:
        return '';
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100 font-sans">
      <header className="p-4 text-center border-b border-gray-700">
        <h1 className="text-2xl font-bold text-teal-400">Gemini Live Audio Chat</h1>
        <p className="text-sm text-gray-400">Real-time voice conversation with AI</p>
      </header>

      <main className="flex-1 flex flex-col p-4 overflow-hidden">
        <div className="flex-1 overflow-y-auto pr-2 space-y-4">
          {transcript.map((entry, index) => (
            <div key={index} className={`flex items-start gap-3 ${entry.speaker === 'user' ? 'justify-end' : 'justify-start'}`}>
              {entry.speaker === 'gemini' && <GeminiIcon />}
              <div className={`max-w-xs md:max-w-md lg:max-w-2xl px-4 py-2 rounded-2xl ${entry.speaker === 'user' ? 'bg-teal-600 rounded-br-none' : 'bg-gray-700 rounded-bl-none'}`}>
                <p className="text-base">{entry.text}</p>
              </div>
              {entry.speaker === 'user' && <UserIcon />}
            </div>
          ))}
           {transcript.length === 0 && status !== ConversationStatus.IDLE && (
            <div className="text-center text-gray-500 pt-10">
              Start speaking to see the transcript...
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>

        {error && <div className="text-center text-red-500 p-2">{error}</div>}
      </main>

      <footer className="p-4 flex flex-col items-center justify-center border-t border-gray-700 bg-gray-900/80 backdrop-blur-sm">
        <button
          onClick={handleButtonClick}
          disabled={status === ConversationStatus.PROCESSING}
          className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-opacity-50
            ${status === ConversationStatus.LISTENING ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500' : 'bg-teal-500 hover:bg-teal-600 focus:ring-teal-400'}
            ${status === ConversationStatus.PROCESSING ? 'bg-gray-500 cursor-not-allowed' : ''}
          `}
        >
          {status === ConversationStatus.PROCESSING && <LoadingSpinnerIcon />}
          {status === ConversationStatus.IDLE && <MicrophoneIcon />}
          {status === ConversationStatus.LISTENING && <StopIcon />}
          {status === ConversationStatus.LISTENING && <span className="absolute h-full w-full rounded-full bg-red-600 animate-ping opacity-75"></span>}
        </button>
        <p className="mt-3 text-sm text-gray-400 h-5">{getStatusText()}</p>
      </footer>
    </div>
  );
};

export default App;
