
export enum ConversationStatus {
  IDLE = 'idle',
  LISTENING = 'listening',
  PROCESSING = 'processing'
}

export interface TranscriptEntry {
  speaker: 'user' | 'gemini';
  text: string;
}
