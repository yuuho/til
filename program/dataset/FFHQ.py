import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tqdm
import numpy as np
import cv2
import torch
import tensorflow as tf


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir_path, tmp_dir_path, **params):
        super().__init__()

        # tfrecord はランダムアクセスできないのでファイルとして展開する場所を作る
        self.tmp_records_path = tmp_dir_path / 'dataset' / params['temp_prefix']
        self.tmp_records_path.mkdir(parents=True,exist_ok=True)

        # 読み込むファイル
        self.tfrecord_path = dataset_dir_path / params['tfrecord_path']
        # tfrecord データセットをロード
        dataset = tf.data.TFRecordDataset( str(self.tfrecord_path) )

        # データセット内のレコード個数を数える
        print('count records...')
        self.length = sum(1 for _ in dataset)
        print(self.length, 'records exists.')
        
        # 各レコードの保存場所を決める
        # for example ) idx -> Path( '/path/to/tmp/dir/%05d.npy'%(idx) )
        self.filepath = lambda idx: self.tmp_records_path / ( ('%0'+str(int(np.ceil(np.log10(self.length))))+'d.npy')%idx )
        
        # レコードの大きさを見る
        self.shape = tuple([ ( ex.ParseFromString(d.numpy()), ex.features.feature['shape'].int64_list.value )[1]
                        for ex in [tf.train.Example()] for d in dataset.take(1)][0])
        print('image size :', self.shape)
        
        if False:
            # 全データを舐める
            load_pbar = tqdm.tqdm(total=self.length)
            load_pbar.set_description("tfrecord loads ")
            for i, data in dataset.enumerate():
                load_pbar.update(1)

                # index を数値にする
                idx = i.numpy()
                # 既に存在したら無視して次に行く
                if self.filepath(idx).exists(): continue
                
                # tf.Example を作成
                ex = tf.train.Example(); ex.ParseFromString( data.numpy() )
                # 画像を ndarray uint8 (C,H,W) で取得して保存
                img = np.fromstring( ex.features.feature['data'].bytes_list.value[0], np.uint8).reshape(self.shape)
                np.save( self.filepath(idx), img )
            load_pbar.close()
        
        self.mode = params['mode']
        self.num_train = int( self.length * params['train_rate'] )
        print('dataset num:',len(self))
    
    def __len__(self):
        return self.num_train if self.mode=='train' else self.length-self.num_train

    def __getitem__(self,idx):
        path = self.filepath(idx if self.mode=='train' else self.num_train+idx )
        img = np.load(path).astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.Tensor(img)
