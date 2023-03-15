## Projects

### 1. [Stanford NLP Lecture Transcription using OpenAI's Whisper](http://3.14.28.154/)

Whisper is an automatic speech recognition (ASR) model trained on hours of multilingual and multitask supervised data. It is implemented as an encoder-decoder transformer architecture where audio are splitted into 30 seconds of chunks, converted into a log-Mel spectrogram, and then passed into an encoder. The decoder is trained to predict the corresponding text caption, intermixed with special tokens that direct the single model to perform tasks such as language identification, phrase-level timestamps, multilingual speech transcription, and to-English speech translation. For more info about whisper, read [here](https://openai.com/blog/whisper/).

I used whisper model to transcribe Stanford NLP lectures into corresponding text captions. [Here](http://3.14.28.154/) is the result of the transcribed lectures. This web app is build using Flask and deployed on AWS EC2 instance. You can find transcribed audio file in the form of text [here](https://github.com/nepalprabin/whisper-webapp/blob/main/Stanford_NLP_lecture_transcripts.zip).

## 2. [Custom Named Entity Recognizer for clinical data](https://nepalprabin-clinical-ner-visualizer-jb4wt5.streamlit.app/)

Named Entity Recognition (NER) is a subtask of Natural Language Processing (NLP) that involves identifying and categorizing named entities in text.

I have developed a custom named entity recognition (NER) model for clinical data using the spacy framework and deployed it using Streamlit.
The model is capable of identifying various entities such as diseases, treatments, medications, and anatomical locations from clinical text data.
The model classifies entities based on three classes: <code>'MEDICINE'</code>, <code>"MEDICALCONDITION"</code>, and <code>"PATHOGEN"</code>. The dataset was used from [kaggle](https://www.kaggle.com/datasets/finalepoch/medical-ner).
You can try the application on this [link](https://nepalprabin-clinical-ner-visualizer-jb4wt5.streamlit.app/)