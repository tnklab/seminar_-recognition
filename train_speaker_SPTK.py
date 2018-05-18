# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import chainer.datasets as D
from chainer import training
from chainer.training import extensions

class DNN(chainer.Chain):
    def __init__(self, n_out):        
        super(DNN, self).__init__(                        
            l1=L.Linear(None, 1024),
            l2=L.Linear(None, 1024),
            # l3=L.Linear(None, 1024),
            # l4=L.Linear(None, 1024),
            l5=L.Linear(None, n_out),
        )

    # DNNのフォワード処理
    def __call__(self, x):
        #h = F.tanh(self.l1(x))
        #h = F.tanh(self.l2(h))
        #h = F.tanh(self.l3(h))
        #h = F.tanh(self.l4(h))
        h = F.dropout(F.tanh(self.l1(x)), ratio=0.5)
        h = F.dropout(F.tanh(self.l2(h)), ratio=0.5)
        # h = F.dropout(F.tanh(self.l3(h)), ratio=0.5)
        # h = F.dropout(F.tanh(self.l4(h)), ratio=0.5)
        return self.l5(h)

def main():
    
    # オプション処理
    parser = argparse.ArgumentParser(description='話者認識モデルの学習')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--datadir', '-d', default='train',
                        help='学習データのディレクトリ')
    args = parser.parse_args()

    sys.stderr.write('GPU: {}\n'.format(args.gpu))
    sys.stderr.write('# minibatch-size: {}\n'.format(args.batchsize))
    sys.stderr.write('# epoch: {}\n'.format(args.epoch))

    trainf = []
    label = 0
    print('loading dataset')
    mfcc = os.listdir(args.datadir)
    for i in [f for f in mfcc if ('mfcc' in f)]:
        trainf.append([os.path.join(args.datadir, i), label])            
        label += 1
    #print('{}'.format(trainf))
    input = []
    target = []
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    log = open(args.out+"/class_log",'w')
    for file in trainf:
        print('{}'.format(file))
        log.write('{},{}\n'.format(file[0].strip().split("/")[-1].split(".")[0],file[1]))
        with open(file[0], 'r') as f:
            lines = f.readlines()         
        for i, l in enumerate(lines):
            #print('{}'.format(len(lines)))
            tmp = []
            flag = False
            for j in range(3): # 3フレームで評価
                if i + 2 < len(lines):
                    frame = lines[i+j].strip().split(" ")
                    #print("i:{},file:{}".format(i,file))
                    np.array(frame,dtype=np.float32)
                    tmp.extend(frame)
                    flag = True
                    #print('{}'.format(tmp))
            if flag:
                input.append(tmp)
                target.append(file[1])
            #    print('{}'.format(input))
    log.close()
    #print('{}'.format(input))  
    #sys.exit()
    #print(np.array(input))
    input = np.array(input).astype(np.float32)
    target = np.array(target).astype(np.int32)
    #print('{},{}'.format(len(input), len(target)))
    train = D.TupleDataset(input, target)
    #sys.stderr.write(train)
    #print(len(input)*0.9)
    train, test = D.split_dataset_random(train, int(len(input)*0.9))
    print('{},{}'.format(len(train), len(test)))
    #sys.exit()
        
    model = L.Classifier(DNN(label)) # CNNにする
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    # trainer.extend(
    #     extensions.PlotReport('main/loss', 'epoch', file_name='loss.png'))
    # trainer.extend(
    #     extensions.PlotReport('main/accuracy', 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    #trainer.extend(extensions.PrintReport(
    #    ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    print('training start!')
    trainer.run()
    # モデルをCPU対応へ
    model.to_cpu()
    # 保存
    modelname = args.out + "/speaker.model"
    print('save the trained model: {}'.format(modelname))
    chainer.serializers.save_npz(modelname, model)

if __name__ == '__main__':
    main()
