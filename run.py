#!/usr/bin/env python
import SSR
import sys

ac = SSR.AudioCorrection(sys.argv[1], 'tfSessions/2018-10-08-12:57:33-0.8590971/session.ckpt')
ac.process()
ac.saveCorrectedAudio()

# Better to run in IDE