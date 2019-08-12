#!/usr/bin/env python
import librosa
import sys

y, sr = librosa.load(sys.argv[1])
y_16k = librosa.resample(y, sr, 16000)
librosa.output.write_wav(sys.argv[0] + "-resampled.wav", y_16k, 16000)
