#!/usr/bin/env python
import MADRSS
import sys

ac =MADRSS.AudioCorrection(sys.argv[1], 'tfSessions/2018-10-08-12:57:33-0.8590971/session.ckpt')
ac.process()
ac.saveCorrectedAudio()

# Better to run in VSCODE
