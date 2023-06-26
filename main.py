from flask import Flask, render_template, redirect, url_for, request, flash
from flask_bootstrap import Bootstrap

import sys

project_name='Real-Time-Voice-Cloning'

sys.path.append(project_name)

from IPython.display import display, Audio, clear_output
from IPython.utils import io
import ipywidgets as widgets
import numpy as np

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path

encoder.load_model(project_name / Path("saved_models/default/encoder.pt"))
# synthesizer = Synthesizer(project_name / Path("synthetizer/models/pretrained.pt"))
synthesizer = Synthesizer("Real-Time-Voice-Cloning/synthesizer/models/pretrained.pt")
vocoder.load_model(project_name / Path("saved_models/default/vocoder.pt"))

SAMPLE_RATE = 22050

embedding = None
def _compute_embedding(audio):
  display(Audio(audio, rate=SAMPLE_RATE, autoplay=True))
  global embedding
  embedding = None
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))

def _upload_audio(b):
  clear_output()
  # audio = upload_audio(sample_rate=SAMPLE_RATE)
  _compute_embedding(b)

# _upload_audio("apjmain.mp3")

# text = "Here is what I told them to help them learn now, rather than later. For, every mistake we make in our lives has a cost. Awareness prevents or may reduce any resultant suffering."  # @param {type:"string"}


def synthesize(embed, text):
  print("Synthesizing new audio...")
  # with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
  clear_output()
  req=Audio(generated_wav, rate=synthesizer.sample_rate, autoplay=True)
  print(req)
  with open('static/final.mp3', 'wb') as f:
    f.write(req.data)

# synthesize(embedding, text)



app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = '49itk5krtgktpr5ktgprkyh'

@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")

@app.route('/submission', methods=['POST'])
def submission():
    req_text = request.form['req_text']
    fighters=request.form['fighters']
    if fighters == 'APJ Abdul Kalam':
      _upload_audio("Audio/apjmain.mp3")
    if fighters == 'Mahatma Gandhi':
      _upload_audio("Audio/gandhi.mp4")
    if fighters == 'APJ Abdul Kalam':
      _upload_audio("Audio/patel.wav")
    synthesize(embedding, req_text)
    return render_template("result.html")

if __name__ == "__main__":
    app.run()