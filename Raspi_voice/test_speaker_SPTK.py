# -*- coding: utf-8 -*-
#
# Chainer 2.1.0
#
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
#import cupy as cp
import commands
import re
import collections

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import chainer.datasets as D
from chainer import training
from chainer.training import extensions

import wave
import pyaudio


class DNN(chainer.Chain):
    def __init__(self, n_out):
        super(DNN, self).__init__(
            l1=L.Linear(None, 1024),
            l2=L.Linear(None, 1024),
            #l3=L.Linear(None, 1024),
            #l4=L.Linear(None, 1024),
            l5=L.Linear(None, n_out),
        )

    # DNNのフォワード処理
    def __call__(self, x):
        h = F.dropout(F.tanh(self.l1(x)), ratio=0.0)
        h = F.dropout(F.tanh(self.l2(h)), ratio=0.0)
        #h = F.dropout(F.tanh(self.l3(h)), ratio=0.0)
        #h = F.dropout(F.tanh(self.l4(h)), ratio=0.0)
        return self.l5(h)


def hantei(wavfile,args,model):
    #　話者認識開始
    print('Calculating MFCC ')
    #cmd = 'compute-mfcc-feats --config=mfcc.conf scp,p:' + args.wavscp + ' ark:- | add-deltas ark:- ark,t:- | perl normalize.pl'
    
    cmd = "cat " + wavfile + \
        " | x2x +sf | frame -l 640 -p 160 | mfcc -l 640 -f 16 -m 13 -n 20 -a 0.97 | python convert_text_mfcc.py |perl normalize_SPTK_modified_print.pl| sed 's/\t/ /g'"

    status, output = commands.getstatusoutput(cmd)  # MFCC計算
    #print(str(output))
    output = output.split('\n')

    input = []
    for i, l in enumerate(output):
        tmp = []
        flag = False
        if not re.match(r"[a-zA-Z]+", l):  # ログを排除
            for j in range(3):  # 3フレームで評価
                if i + 2 < len(output):
                    frame = output[i + j].strip().split(" ")
                    tmp.extend(frame)
                    flag = True
                    # print('{}'.format(tmp))
            if flag:
                input.append(tmp)
        # else:
        #    print('m {}'.format(l))
    # print('{}'.format(input))
    input = np.array(input).astype(np.float32)
    # print('{}'.format(input))
    # sys.exit()

    

    # x = chainer.Variable(np.asarray(input))
    y = model.predictor(chainer.Variable(np.asarray(input)))
    #c = F.softmax(y).data.argmax()
    c = F.argmax(F.softmax(y), axis=1)
    count = collections.Counter(c.data)
    print('{}'.format(count))
    speaker = count.most_common(1)[0][0]
    # print('result {}'.format(speaker))
    total = 0
    for t in count: # 総フレーム数の計算
        total += count[t]
    #print('total {}'.format(total))
    speaker = count.most_common(1)[0][0]
    conf = float(count.most_common(1)[0][1]) / total # 信頼度計算        
    s_msg = str(speaker) if conf >= args.confidence else '-1' # 0.9以上なら確定        
    print('result {}, 1st candidate {}, conf {}'.format(s_msg, speaker, conf))


def main():
    # 録音環境設定
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024*2
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "speaker.wav"

    # オプション処理
    parser = argparse.ArgumentParser(description='話者認識')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m',type=str,default='model',
                        help='モデル')
    parser.add_argument("--wavdata", "-w", type=str,
                        default="", help="WAVファイル")
    parser.add_argument('--confidence', '-c', type=float, default=0.9,
    help='信頼度')
    parser.add_argument('--number', '-n', type=int, default=6,
    help='話者数')
    args = parser.parse_args()

    model = L.Classifier(DNN(args.number))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # load model
    chainer.serializers.load_npz(args.model, model)

    # 録音
    audio = pyaudio.PyAudio()
    print(args.model.strip().split("/")[:-1])
    #sys.exit(1)
    while True:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=3,
                            frames_per_buffer=CHUNK)
        print("recording")
        frames = []
        for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.append(data)
        print("record finish")
        stream.stop_stream()
        stream.close()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME,"wb")
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(16000)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        #sys.exit(1)
        hantei(WAVE_OUTPUT_FILENAME,args,model)

    


if __name__ == '__main__':
    main()
