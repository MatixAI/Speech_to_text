# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
import deepspeech
import wave
import numpy as np
print("Kindly Select the model you want to use for translation\n 1. \tQuartzNet\n 2. \tDeepSpeech")
x=input();
if x=='1':
  quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En");
  files = ['/content/Welcome.wav'];
  for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
    text_file = open("/content/translation.txt", "w")
    text_file.write("The audio translation is: %s" % transcription)
    text_file.close()
    print(f"Audio in {fname} was recognized as: {transcription}");
elif x=='2':
  model_file_path = 'deepspeech-0.8.2-models/deepspeech-0.8.1-models.pbmm'
  model = deepspeech.Model(model_file_path)
  
  scorer_file_path = 'deepspeech-0.8.2-models/deepspeech-0.8.1-models.scorer'
  model.enableExternalScorer(scorer_file_path)
  lm_alpha = 0.75
  lm_beta = 1.85
  model.setScorerAlphaBeta(lm_alpha, lm_beta)
  beam_width = 500
  model.setBeamWidth(beam_width)

  filename = 'audio/8455-210777-0068.wav'
  w = wave.open(filename, 'r')
  rate = w.getframerate()
  frames = w.getnframes()
  buffer = w.readframes(frames)
  data16 = np.frombuffer(buffer, dtype=np.int16)
  text = model.stt(data16)
  text_file = open("/content/translation.txt", "w")
  text_file.write("The audio translation is: %s" % text)
  text_file.close()
  print("Not equal")