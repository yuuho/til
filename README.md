# gans

```
conda create -n pytorch16 numpy scipy scikit-image scikit-learn matplotlib pandas \
                        opencv tqdm pyyaml jupyterlab tensorboard \
                        pytorch torchvision cudatoolkit -c pytorch
conda activate pytorch16
pip install git+https://github.com/yuuho/Hellflame
```

```
conda install tensorflow-datasets
```


## 環境構築

```
conda create -n pytorch16 numpy scipy scikit-image scikit-learn matplotlib pandas \
                        opencv tqdm pyyaml jupyterlab tensorboard tensorflow-datasets \
                        pytorch torchvision cudatoolkit -c pytorch
conda activate pytorch16
pip install git+https://github.com/yuuho/Hellflame
```



## データセット
Style-GANの論文で作成されたデータセット
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

Google driveに用意されているものをダウンロードしてきた．

```

nas:/path/to/FFHQ/ffhq-dataset/
    - ffhq-dataset-v2.json
    - images1024x1024/
    - LICENSE.txt
    - README.txt
    - tfrecords/ffhq/
        - ffhq-r02.tfrecords
        - ffhq-r03.tfrecords
        ...
    - zips/


machine:$MLDATA/tf_ffhq/ffhq
    - 


```

