from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
import torch
import soundfile as sf

import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

string=str(input("Enter text: "))
inputs = processor(text=string, return_tensors="pt")

#embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

#audio_files = glob.glob('E:\Speech\LJSpeech-1.1\wavs' + "/*.wav")

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
#signal,fs=torchaudio.load('Aayush-1.wav')
signal,fs=torchaudio.load('22k_denoised.wav')

#signal,fs=torchaudio.load('LJSpeech-1.1\wavs\LJ001-0010.wav')
print(fs)
if (fs != 16000):
    signal = torchaudio.transforms.Resample(fs, 16000)(signal)
speaker_embeddings = classifier.encode_batch(signal)
speaker_embeddings = F.normalize(speaker_embeddings, dim=2)
speaker_embeddings=torch.squeeze(speaker_embeddings,0)
print(speaker_embeddings)

#speaker_embeddings = torch.tensor(embeddings_dataset[6000]["xvector"]).unsqueeze(0)

spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

with torch.no_grad():
    speech = vocoder(spectrogram)
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("tts_example2.wav", speech.numpy(), samplerate=16000)