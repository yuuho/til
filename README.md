# GANs & cGANs

GAN の再現実装をしつつ性能調査をしつつ

## 環境構築

```
conda create -n pytorch16 numpy scipy scikit-image scikit-learn matplotlib pandas \
                        opencv tqdm pyyaml jupyterlab tensorboard tensorflow \
                        pytorch torchvision cudatoolkit -c pytorch
conda activate pytorch16
pip install git+https://github.com/yuuho/Hellflame
```


## データセット
Style-GANの論文で作成されたデータセット
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
を使う．

Google driveに用意されている FFHQ データセットをすべてダウンロードしてきた．
NAS に置いてある．ローカルの machine にコピーして使う．

NAS側 )
```
NAS:/path/to/FFHQ/ffhq-dataset/
    - ffhq-dataset-v2.json
    - images1024x1024/
    - LICENSE.txt
    - README.txt
    - tfrecords/ffhq/           }
        - ffhq-r02.tfrecords    }
        - ffhq-r03.tfrecords    }
        ...                     }
        - ffhq-r10.tfrecords    }
        - LICENSE.txt           }
    - zips/
```

ローカルマシン側 )
```
machine:$MLDATA/tf_ffhq/ffhq    }
    - ffhq-r02.tfrecords        }
    - ffhq-r03.tfrecords        }
    ...                         }
    - ffhq-r10.tfrecords        }
    - LICENSE.txt               }
```

