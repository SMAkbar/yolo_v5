
# YOLO_V5 IMPLEMENTATION FOR OBJECT DETECTION


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
!git clone https://github.com/ultralytics/yolov5
!cd yolov5
!pip install -r yolov5/requirements.txt
```

    fatal: destination path 'yolov5' already exists and is not an empty directory.
    Requirement already satisfied: matplotlib>=3.2.2 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 4)) (3.2.2)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 5)) (1.19.5)
    Requirement already satisfied: opencv-python>=4.1.2 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 6)) (4.1.2.30)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 7)) (7.1.2)
    Collecting PyYAML>=5.3.1
    [?25l  Downloading https://files.pythonhosted.org/packages/7a/a5/393c087efdc78091afa2af9f1378762f9821c9c1d7a22c5753fb5ac5f97a/PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636kB)
    [K     |████████████████████████████████| 645kB 18.4MB/s 
    [?25hRequirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 9)) (1.4.1)
    Requirement already satisfied: torch>=1.7.0 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 10)) (1.8.1+cu101)
    Requirement already satisfied: torchvision>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 11)) (0.9.1+cu101)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 12)) (4.41.1)
    Requirement already satisfied: tensorboard>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 15)) (2.4.1)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 19)) (0.11.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 20)) (1.1.5)
    Collecting thop
      Downloading https://files.pythonhosted.org/packages/6c/8b/22ce44e1c71558161a8bd54471123cc796589c7ebbfc15a7e8932e522f83/thop-0.0.31.post2005241907-py3-none-any.whl
    Requirement already satisfied: pycocotools>=2.0 in /usr/local/lib/python3.7/dist-packages (from -r yolov5/requirements.txt (line 29)) (2.0.2)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r yolov5/requirements.txt (line 4)) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r yolov5/requirements.txt (line 4)) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r yolov5/requirements.txt (line 4)) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.2->-r yolov5/requirements.txt (line 4)) (1.3.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.7.0->-r yolov5/requirements.txt (line 10)) (3.7.4.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (2.23.0)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.12.4)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.15.0)
    Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (0.36.2)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (0.4.3)
    Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.32.0)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.28.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.0.1)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (54.2.0)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (0.12.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.8.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.3.4)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->-r yolov5/requirements.txt (line 20)) (2018.9)
    Requirement already satisfied: cython>=0.27.3 in /usr/local/lib/python3.7/dist-packages (from pycocotools>=2.0->-r yolov5/requirements.txt (line 29)) (0.29.22)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (2020.12.5)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (2.10)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (1.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3.6" in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (4.2.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (0.2.8)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.8.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.1.0)
    Requirement already satisfied: pyasn1>=0.1.3 in /usr/local/lib/python3.7/dist-packages (from rsa<5,>=3.1.4; python_version >= "3.6"->google-auth<2,>=1.6.3->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (0.4.8)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard>=2.4.1->-r yolov5/requirements.txt (line 15)) (3.4.1)
    Installing collected packages: PyYAML, thop
      Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
    Successfully installed PyYAML-5.4.1 thop-0.0.31.post2005241907
    


```python
!wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
```

    --2021-04-01 17:26:20--  https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
    Resolving github.com (github.com)... 140.82.113.3
    Connecting to github.com (github.com)|140.82.113.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/264818686/68120200-4399-11eb-875d-220ea1c09e18?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210401%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210401T172620Z&X-Amz-Expires=300&X-Amz-Signature=008beb5f9e21e72e32688244e5a8d9fc10a32a93b80b839e2339ab0482953cfe&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dcoco128.zip&response-content-type=application%2Foctet-stream [following]
    --2021-04-01 17:26:20--  https://github-releases.githubusercontent.com/264818686/68120200-4399-11eb-875d-220ea1c09e18?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210401%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210401T172620Z&X-Amz-Expires=300&X-Amz-Signature=008beb5f9e21e72e32688244e5a8d9fc10a32a93b80b839e2339ab0482953cfe&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dcoco128.zip&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.109.154, 185.199.108.154, 185.199.110.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.109.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 22091032 (21M) [application/octet-stream]
    Saving to: ‘coco128.zip’
    
    coco128.zip         100%[===================>]  21.07M  --.-KB/s    in 0.1s    
    
    2021-04-01 17:26:20 (161 MB/s) - ‘coco128.zip’ saved [22091032/22091032]
    
    


```python
!unzip coco128.zip
```

    Archive:  coco128.zip
       creating: coco128/
      inflating: coco128/.DS_Store       
      inflating: coco128/LICENSE         
       creating: coco128/images/
      inflating: coco128/images/.DS_Store  
       creating: coco128/images/train2017/
      inflating: coco128/images/train2017/000000000612.jpg  
      inflating: coco128/images/train2017/000000000404.jpg  
      inflating: coco128/images/train2017/000000000438.jpg  
      inflating: coco128/images/train2017/000000000389.jpg  
      inflating: coco128/images/train2017/000000000564.jpg  
      inflating: coco128/images/train2017/000000000149.jpg  
      inflating: coco128/images/train2017/000000000605.jpg  
      inflating: coco128/images/train2017/000000000349.jpg  
      inflating: coco128/images/train2017/000000000201.jpg  
      inflating: coco128/images/train2017/000000000599.jpg  
      inflating: coco128/images/train2017/000000000572.jpg  
      inflating: coco128/images/train2017/000000000360.jpg  
      inflating: coco128/images/train2017/000000000370.jpg  
      inflating: coco128/images/train2017/000000000562.jpg  
      inflating: coco128/images/train2017/000000000589.jpg  
      inflating: coco128/images/train2017/.DS_Store  
      inflating: coco128/images/train2017/000000000359.jpg  
      inflating: coco128/images/train2017/000000000629.jpg  
      inflating: coco128/images/train2017/000000000165.jpg  
      inflating: coco128/images/train2017/000000000415.jpg  
      inflating: coco128/images/train2017/000000000575.jpg  
      inflating: coco128/images/train2017/000000000560.jpg  
      inflating: coco128/images/train2017/000000000400.jpg  
      inflating: coco128/images/train2017/000000000428.jpg  
      inflating: coco128/images/train2017/000000000164.jpg  
      inflating: coco128/images/train2017/000000000315.jpg  
      inflating: coco128/images/train2017/000000000077.jpg  
      inflating: coco128/images/train2017/000000000089.jpg  
      inflating: coco128/images/train2017/000000000260.jpg  
      inflating: coco128/images/train2017/000000000328.jpg  
      inflating: coco128/images/train2017/000000000472.jpg  
      inflating: coco128/images/train2017/000000000510.jpg  
      inflating: coco128/images/train2017/000000000074.jpg  
      inflating: coco128/images/train2017/000000000049.jpg  
      inflating: coco128/images/train2017/000000000061.jpg  
      inflating: coco128/images/train2017/000000000263.jpg  
      inflating: coco128/images/train2017/000000000459.jpg  
      inflating: coco128/images/train2017/000000000471.jpg  
      inflating: coco128/images/train2017/000000000307.jpg  
      inflating: coco128/images/train2017/000000000529.jpg  
      inflating: coco128/images/train2017/000000000071.jpg  
      inflating: coco128/images/train2017/000000000064.jpg  
      inflating: coco128/images/train2017/000000000514.jpg  
      inflating: coco128/images/train2017/000000000474.jpg  
      inflating: coco128/images/train2017/000000000312.jpg  
      inflating: coco128/images/train2017/000000000110.jpg  
      inflating: coco128/images/train2017/000000000138.jpg  
      inflating: coco128/images/train2017/000000000338.jpg  
      inflating: coco128/images/train2017/000000000502.jpg  
      inflating: coco128/images/train2017/000000000072.jpg  
      inflating: coco128/images/train2017/000000000073.jpg  
      inflating: coco128/images/train2017/000000000488.jpg  
      inflating: coco128/images/train2017/000000000113.jpg  
      inflating: coco128/images/train2017/000000000136.jpg  
      inflating: coco128/images/train2017/000000000650.jpg  
      inflating: coco128/images/train2017/000000000446.jpg  
      inflating: coco128/images/train2017/000000000308.jpg  
      inflating: coco128/images/train2017/000000000491.jpg  
      inflating: coco128/images/train2017/000000000532.jpg  
      inflating: coco128/images/train2017/000000000283.jpg  
      inflating: coco128/images/train2017/000000000042.jpg  
      inflating: coco128/images/train2017/000000000081.jpg  
      inflating: coco128/images/train2017/000000000094.jpg  
      inflating: coco128/images/train2017/000000000241.jpg  
      inflating: coco128/images/train2017/000000000490.jpg  
      inflating: coco128/images/train2017/000000000309.jpg  
      inflating: coco128/images/train2017/000000000321.jpg  
      inflating: coco128/images/train2017/000000000109.jpg  
      inflating: coco128/images/train2017/000000000486.jpg  
      inflating: coco128/images/train2017/000000000531.jpg  
      inflating: coco128/images/train2017/000000000257.jpg  
      inflating: coco128/images/train2017/000000000294.jpg  
      inflating: coco128/images/train2017/000000000450.jpg  
      inflating: coco128/images/train2017/000000000322.jpg  
      inflating: coco128/images/train2017/000000000326.jpg  
      inflating: coco128/images/train2017/000000000332.jpg  
      inflating: coco128/images/train2017/000000000508.jpg  
      inflating: coco128/images/train2017/000000000520.jpg  
      inflating: coco128/images/train2017/000000000078.jpg  
      inflating: coco128/images/train2017/000000000086.jpg  
      inflating: coco128/images/train2017/000000000092.jpg  
      inflating: coco128/images/train2017/000000000247.jpg  
      inflating: coco128/images/train2017/000000000643.jpg  
      inflating: coco128/images/train2017/000000000133.jpg  
      inflating: coco128/images/train2017/000000000127.jpg  
      inflating: coco128/images/train2017/000000000641.jpg  
      inflating: coco128/images/train2017/000000000443.jpg  
      inflating: coco128/images/train2017/000000000536.jpg  
      inflating: coco128/images/train2017/000000000250.jpg  
      inflating: coco128/images/train2017/000000000196.jpg  
      inflating: coco128/images/train2017/000000000357.jpg  
      inflating: coco128/images/train2017/000000000431.jpg  
      inflating: coco128/images/train2017/000000000419.jpg  
      inflating: coco128/images/train2017/000000000394.jpg  
      inflating: coco128/images/train2017/000000000009.jpg  
      inflating: coco128/images/train2017/000000000034.jpg  
      inflating: coco128/images/train2017/000000000544.jpg  
      inflating: coco128/images/train2017/000000000395.jpg  
      inflating: coco128/images/train2017/000000000626.jpg  
      inflating: coco128/images/train2017/000000000154.jpg  
      inflating: coco128/images/train2017/000000000142.jpg  
      inflating: coco128/images/train2017/000000000368.jpg  
      inflating: coco128/images/train2017/000000000397.jpg  
      inflating: coco128/images/train2017/000000000208.jpg  
      inflating: coco128/images/train2017/000000000036.jpg  
      inflating: coco128/images/train2017/000000000584.jpg  
      inflating: coco128/images/train2017/000000000590.jpg  
      inflating: coco128/images/train2017/000000000382.jpg  
      inflating: coco128/images/train2017/000000000194.jpg  
      inflating: coco128/images/train2017/000000000625.jpg  
      inflating: coco128/images/train2017/000000000143.jpg  
      inflating: coco128/images/train2017/000000000581.jpg  
      inflating: coco128/images/train2017/000000000595.jpg  
      inflating: coco128/images/train2017/000000000542.jpg  
      inflating: coco128/images/train2017/000000000387.jpg  
      inflating: coco128/images/train2017/000000000436.jpg  
      inflating: coco128/images/train2017/000000000634.jpg  
      inflating: coco128/images/train2017/000000000620.jpg  
      inflating: coco128/images/train2017/000000000636.jpg  
      inflating: coco128/images/train2017/000000000144.jpg  
      inflating: coco128/images/train2017/000000000540.jpg  
      inflating: coco128/images/train2017/000000000597.jpg  
      inflating: coco128/images/train2017/000000000030.jpg  
      inflating: coco128/images/train2017/000000000025.jpg  
      inflating: coco128/images/train2017/000000000569.jpg  
      inflating: coco128/images/train2017/000000000384.jpg  
      inflating: coco128/images/train2017/000000000192.jpg  
      inflating: coco128/images/train2017/000000000623.jpg  
      inflating: coco128/images/train2017/000000000151.jpg  
       creating: coco128/labels/
      inflating: coco128/labels/.DS_Store  
       creating: coco128/labels/train2017/
      inflating: coco128/labels/train2017/000000000260.txt  
      inflating: coco128/labels/train2017/000000000089.txt  
      inflating: coco128/labels/train2017/000000000328.txt  
      inflating: coco128/labels/train2017/000000000472.txt  
      inflating: coco128/labels/train2017/000000000315.txt  
      inflating: coco128/labels/train2017/000000000077.txt  
      inflating: coco128/labels/train2017/000000000263.txt  
      inflating: coco128/labels/train2017/000000000049.txt  
      inflating: coco128/labels/train2017/000000000061.txt  
      inflating: coco128/labels/train2017/000000000459.txt  
      inflating: coco128/labels/train2017/000000000471.txt  
      inflating: coco128/labels/train2017/000000000074.txt  
      inflating: coco128/labels/train2017/000000000510.txt  
      inflating: coco128/labels/train2017/000000000514.txt  
      inflating: coco128/labels/train2017/000000000064.txt  
      inflating: coco128/labels/train2017/000000000110.txt  
      inflating: coco128/labels/train2017/000000000138.txt  
      inflating: coco128/labels/train2017/000000000312.txt  
      inflating: coco128/labels/train2017/000000000474.txt  
      inflating: coco128/labels/train2017/.DS_Store  
      inflating: coco128/labels/train2017/000000000307.txt  
      inflating: coco128/labels/train2017/000000000071.txt  
      inflating: coco128/labels/train2017/000000000529.txt  
      inflating: coco128/labels/train2017/000000000073.txt  
      inflating: coco128/labels/train2017/000000000113.txt  
      inflating: coco128/labels/train2017/000000000488.txt  
      inflating: coco128/labels/train2017/000000000338.txt  
      inflating: coco128/labels/train2017/000000000072.txt  
      inflating: coco128/labels/train2017/000000000502.txt  
      inflating: coco128/labels/train2017/000000000149.txt  
      inflating: coco128/labels/train2017/000000000389.txt  
      inflating: coco128/labels/train2017/000000000404.txt  
      inflating: coco128/labels/train2017/000000000438.txt  
      inflating: coco128/labels/train2017/000000000612.txt  
      inflating: coco128/labels/train2017/000000000564.txt  
      inflating: coco128/labels/train2017/000000000572.txt  
      inflating: coco128/labels/train2017/000000000599.txt  
      inflating: coco128/labels/train2017/000000000360.txt  
      inflating: coco128/labels/train2017/000000000349.txt  
      inflating: coco128/labels/train2017/000000000605.txt  
      inflating: coco128/labels/train2017/000000000201.txt  
      inflating: coco128/labels/train2017/000000000629.txt  
      inflating: coco128/labels/train2017/000000000359.txt  
      inflating: coco128/labels/train2017/000000000370.txt  
      inflating: coco128/labels/train2017/000000000589.txt  
      inflating: coco128/labels/train2017/000000000562.txt  
      inflating: coco128/labels/train2017/000000000560.txt  
      inflating: coco128/labels/train2017/000000000164.txt  
      inflating: coco128/labels/train2017/000000000400.txt  
      inflating: coco128/labels/train2017/000000000428.txt  
      inflating: coco128/labels/train2017/000000000415.txt  
      inflating: coco128/labels/train2017/000000000165.txt  
      inflating: coco128/labels/train2017/000000000575.txt  
      inflating: coco128/labels/train2017/000000000544.txt  
      inflating: coco128/labels/train2017/000000000034.txt  
      inflating: coco128/labels/train2017/000000000626.txt  
      inflating: coco128/labels/train2017/000000000154.txt  
      inflating: coco128/labels/train2017/000000000395.txt  
      inflating: coco128/labels/train2017/000000000394.txt  
      inflating: coco128/labels/train2017/000000000431.txt  
      inflating: coco128/labels/train2017/000000000357.txt  
      inflating: coco128/labels/train2017/000000000419.txt  
      inflating: coco128/labels/train2017/000000000196.txt  
      inflating: coco128/labels/train2017/000000000009.txt  
      inflating: coco128/labels/train2017/000000000584.txt  
      inflating: coco128/labels/train2017/000000000590.txt  
      inflating: coco128/labels/train2017/000000000143.txt  
      inflating: coco128/labels/train2017/000000000625.txt  
      inflating: coco128/labels/train2017/000000000194.txt  
      inflating: coco128/labels/train2017/000000000382.txt  
      inflating: coco128/labels/train2017/000000000397.txt  
      inflating: coco128/labels/train2017/000000000368.txt  
      inflating: coco128/labels/train2017/000000000142.txt  
      inflating: coco128/labels/train2017/000000000036.txt  
      inflating: coco128/labels/train2017/000000000208.txt  
      inflating: coco128/labels/train2017/000000000542.txt  
      inflating: coco128/labels/train2017/000000000581.txt  
      inflating: coco128/labels/train2017/000000000595.txt  
      inflating: coco128/labels/train2017/000000000634.txt  
      inflating: coco128/labels/train2017/000000000620.txt  
      inflating: coco128/labels/train2017/000000000436.txt  
      inflating: coco128/labels/train2017/000000000387.txt  
      inflating: coco128/labels/train2017/000000000569.txt  
      inflating: coco128/labels/train2017/000000000025.txt  
      inflating: coco128/labels/train2017/000000000623.txt  
      inflating: coco128/labels/train2017/000000000151.txt  
      inflating: coco128/labels/train2017/000000000192.txt  
      inflating: coco128/labels/train2017/000000000384.txt  
      inflating: coco128/labels/train2017/000000000636.txt  
      inflating: coco128/labels/train2017/000000000144.txt  
      inflating: coco128/labels/train2017/000000000030.txt  
      inflating: coco128/labels/train2017/000000000597.txt  
      inflating: coco128/labels/train2017/000000000540.txt  
      inflating: coco128/labels/train2017/000000000241.txt  
      inflating: coco128/labels/train2017/000000000094.txt  
      inflating: coco128/labels/train2017/000000000309.txt  
      inflating: coco128/labels/train2017/000000000321.txt  
      inflating: coco128/labels/train2017/000000000490.txt  
      inflating: coco128/labels/train2017/000000000491.txt  
      inflating: coco128/labels/train2017/000000000446.txt  
      inflating: coco128/labels/train2017/000000000308.txt  
      inflating: coco128/labels/train2017/000000000650.txt  
      inflating: coco128/labels/train2017/000000000136.txt  
      inflating: coco128/labels/train2017/000000000081.txt  
      inflating: coco128/labels/train2017/000000000042.txt  
      inflating: coco128/labels/train2017/000000000283.txt  
      inflating: coco128/labels/train2017/000000000532.txt  
      inflating: coco128/labels/train2017/000000000450.txt  
      inflating: coco128/labels/train2017/000000000322.txt  
      inflating: coco128/labels/train2017/000000000486.txt  
      inflating: coco128/labels/train2017/000000000109.txt  
      inflating: coco128/labels/train2017/000000000294.txt  
      inflating: coco128/labels/train2017/000000000257.txt  
      inflating: coco128/labels/train2017/000000000531.txt  
      inflating: coco128/labels/train2017/000000000247.txt  
      inflating: coco128/labels/train2017/000000000086.txt  
      inflating: coco128/labels/train2017/000000000092.txt  
      inflating: coco128/labels/train2017/000000000643.txt  
      inflating: coco128/labels/train2017/000000000326.txt  
      inflating: coco128/labels/train2017/000000000332.txt  
      inflating: coco128/labels/train2017/000000000078.txt  
     extracting: coco128/labels/train2017/000000000508.txt  
      inflating: coco128/labels/train2017/000000000520.txt  
     extracting: coco128/labels/train2017/000000000250.txt  
      inflating: coco128/labels/train2017/000000000536.txt  
      inflating: coco128/labels/train2017/000000000443.txt  
      inflating: coco128/labels/train2017/000000000133.txt  
      inflating: coco128/labels/train2017/000000000641.txt  
      inflating: coco128/labels/train2017/000000000127.txt  
      inflating: coco128/README.txt      
    


```python
from PIL import Image

im = Image.open('/content/coco128/images/train2017/000000000009.jpg')
im.show()
```

# Testing object detection on "640X___" image


```python
from IPython.display import Image
Image('/content/coco128/images/train2017/000000000009.jpg')
```




![jpeg](output_7_0.jpeg)




```python
!python /content/yolov5/detect.py --source /content/coco128/images/train2017 --weights yolov5s.pt --conf 0.25
```

    Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='/content/coco128/images/train2017', update=False, view_img=False, weights=['yolov5s.pt'])
    [31m[1mrequirements:[0m /content/requirements.txt not found, check failed.
    YOLOv5 🚀 v4.0-180-g1148e2e torch 1.8.1+cu101 CUDA:0 (Tesla T4, 15109.75MB)
    
    Fusing layers... 
    Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
    image 1/128 /content/coco128/images/train2017/000000000009.jpg: 480x640 1 bowl, 1 broccoli, 1 dining table, Done. (0.012s)
    image 2/128 /content/coco128/images/train2017/000000000025.jpg: 448x640 2 giraffes, Done. (0.011s)
    image 3/128 /content/coco128/images/train2017/000000000030.jpg: 448x640 1 dining table, 1 vase, Done. (0.010s)
    image 4/128 /content/coco128/images/train2017/000000000034.jpg: 448x640 1 zebra, Done. (0.010s)
    image 5/128 /content/coco128/images/train2017/000000000036.jpg: 640x512 1 person, 1 umbrella, Done. (0.011s)
    image 6/128 /content/coco128/images/train2017/000000000042.jpg: 480x640 1 dog, Done. (0.011s)
    image 7/128 /content/coco128/images/train2017/000000000049.jpg: 640x512 4 persons, 2 horses, 1 potted plant, Done. (0.011s)
    image 8/128 /content/coco128/images/train2017/000000000061.jpg: 512x640 3 persons, 2 elephants, Done. (0.011s)
    image 9/128 /content/coco128/images/train2017/000000000064.jpg: 640x480 2 cars, 1 stop sign, 1 bench, 1 clock, Done. (0.010s)
    image 10/128 /content/coco128/images/train2017/000000000071.jpg: 448x640 4 cars, 2 trains, Done. (0.010s)
    image 11/128 /content/coco128/images/train2017/000000000072.jpg: 640x448 2 giraffes, Done. (0.010s)
    image 12/128 /content/coco128/images/train2017/000000000073.jpg: 640x576 1 motorcycle, Done. (0.012s)
    image 13/128 /content/coco128/images/train2017/000000000074.jpg: 448x640 9 persons, 1 bicycle, 1 dog, Done. (0.011s)
    image 14/128 /content/coco128/images/train2017/000000000077.jpg: 480x640 4 persons, 2 skateboards, Done. (0.010s)
    image 15/128 /content/coco128/images/train2017/000000000078.jpg: 640x640 1 clock, 1 vase, Done. (0.012s)
    image 16/128 /content/coco128/images/train2017/000000000081.jpg: 448x640 1 airplane, Done. (0.010s)
    image 17/128 /content/coco128/images/train2017/000000000086.jpg: 640x512 1 person, 1 bicycle, Done. (0.011s)
    image 18/128 /content/coco128/images/train2017/000000000089.jpg: 480x640 6 knifes, 1 microwave, 1 oven, 1 book, Done. (0.011s)
    image 19/128 /content/coco128/images/train2017/000000000092.jpg: 448x640 1 fork, 1 cake, Done. (0.010s)
    image 20/128 /content/coco128/images/train2017/000000000094.jpg: 448x640 1 car, 1 truck, 1 traffic light, Done. (0.010s)
    image 21/128 /content/coco128/images/train2017/000000000109.jpg: 416x640 7 persons, 1 boat, 2 benchs, Done. (0.011s)
    image 22/128 /content/coco128/images/train2017/000000000110.jpg: 480x640 12 persons, 4 cups, 2 knifes, 2 pizzas, 2 chairs, 1 dining table, Done. (0.011s)
    image 23/128 /content/coco128/images/train2017/000000000113.jpg: 640x416 3 persons, 1 cup, 1 knife, 1 cake, 2 chairs, 1 dining table, Done. (0.010s)
    image 24/128 /content/coco128/images/train2017/000000000127.jpg: 512x640 4 benchs, 2 umbrellas, 1 cup, 1 knife, 1 spoon, 1 cake, 2 potted plants, 1 dining table, 1 book, Done. (0.011s)
    image 25/128 /content/coco128/images/train2017/000000000133.jpg: 480x640 1 teddy bear, Done. (0.010s)
    image 26/128 /content/coco128/images/train2017/000000000136.jpg: 480x640 1 person, 2 giraffes, Done. (0.010s)
    image 27/128 /content/coco128/images/train2017/000000000138.jpg: 576x640 1 sports ball, 1 bowl, 1 potted plant, 1 refrigerator, 1 clock, 6 vases, Done. (0.012s)
    image 28/128 /content/coco128/images/train2017/000000000142.jpg: 640x480 2 bottles, 1 cup, 1 bowl, 6 bananas, 1 dining table, Done. (0.010s)
    image 29/128 /content/coco128/images/train2017/000000000143.jpg: 544x640 8 birds, Done. (0.012s)
    image 30/128 /content/coco128/images/train2017/000000000144.jpg: 480x640 3 giraffes, Done. (0.010s)
    image 31/128 /content/coco128/images/train2017/000000000149.jpg: 448x640 6 persons, 6 kites, Done. (0.010s)
    image 32/128 /content/coco128/images/train2017/000000000151.jpg: 640x480 1 person, 1 stop sign, Done. (0.010s)
    image 33/128 /content/coco128/images/train2017/000000000154.jpg: 640x448 3 zebras, Done. (0.010s)
    image 34/128 /content/coco128/images/train2017/000000000164.jpg: 480x640 6 bottles, 5 wine glasss, 1 cup, 1 chair, 1 microwave, 1 refrigerator, 1 vase, Done. (0.011s)
    image 35/128 /content/coco128/images/train2017/000000000165.jpg: 544x640 2 persons, 1 tie, Done. (0.012s)
    image 36/128 /content/coco128/images/train2017/000000000192.jpg: 480x640 4 persons, 1 baseball bat, Done. (0.011s)
    image 37/128 /content/coco128/images/train2017/000000000194.jpg: 480x640 1 broccoli, 1 pizza, Done. (0.010s)
    image 38/128 /content/coco128/images/train2017/000000000196.jpg: 480x640 5 spoons, 13 bowls, 6 carrots, 1 dining table, Done. (0.010s)
    image 39/128 /content/coco128/images/train2017/000000000201.jpg: 448x640 2 persons, 3 snowboards, Done. (0.010s)
    image 40/128 /content/coco128/images/train2017/000000000208.jpg: 480x640 1 person, Done. (0.010s)
    image 41/128 /content/coco128/images/train2017/000000000241.jpg: 640x480 5 persons, 1 bottle, 2 cups, 1 couch, 1 potted plant, 1 remote, Done. (0.010s)
    image 42/128 /content/coco128/images/train2017/000000000247.jpg: 448x640 2 cars, 2 airplanes, Done. (0.010s)
    image 43/128 /content/coco128/images/train2017/000000000250.jpg: 480x640 Done. (0.010s)
    image 44/128 /content/coco128/images/train2017/000000000257.jpg: 480x640 23 persons, 1 motorcycle, 1 bus, 3 trucks, 2 traffic lights, 1 backpack, Done. (0.010s)
    image 45/128 /content/coco128/images/train2017/000000000260.jpg: 448x640 2 persons, 1 car, 1 bus, 4 suitcases, Done. (0.010s)
    image 46/128 /content/coco128/images/train2017/000000000263.jpg: 640x608 1 elephant, Done. (0.012s)
    image 47/128 /content/coco128/images/train2017/000000000283.jpg: 640x448 1 bottle, 1 wine glass, 2 chairs, 1 dining table, Done. (0.011s)
    image 48/128 /content/coco128/images/train2017/000000000294.jpg: 448x640 1 person, 6 knifes, 2 spoons, 1 microwave, 2 clocks, Done. (0.010s)
    image 49/128 /content/coco128/images/train2017/000000000307.jpg: 480x640 1 car, 1 dog, 1 frisbee, Done. (0.010s)
    image 50/128 /content/coco128/images/train2017/000000000308.jpg: 448x640 5 persons, 2 umbrellas, 4 wine glasss, 1 chair, 1 potted plant, 1 dining table, 1 clock, Done. (0.010s)
    image 51/128 /content/coco128/images/train2017/000000000309.jpg: 640x640 1 cow, Done. (0.012s)
    image 52/128 /content/coco128/images/train2017/000000000312.jpg: 448x640 5 elephants, Done. (0.010s)
    image 53/128 /content/coco128/images/train2017/000000000315.jpg: 448x640 7 persons, 9 umbrellas, 1 bottle, 5 chairs, Done. (0.010s)
    image 54/128 /content/coco128/images/train2017/000000000321.jpg: 480x640 2 hot dogs, 1 dining table, Done. (0.011s)
    image 55/128 /content/coco128/images/train2017/000000000322.jpg: 512x640 1 person, 2 frisbees, 1 baseball bat, Done. (0.011s)
    image 56/128 /content/coco128/images/train2017/000000000326.jpg: 448x640 1 person, 1 tennis racket, Done. (0.010s)
    image 57/128 /content/coco128/images/train2017/000000000328.jpg: 512x640 3 persons, 1 car, 1 backpack, 2 ties, 1 cell phone, 1 book, Done. (0.011s)
    image 58/128 /content/coco128/images/train2017/000000000332.jpg: 480x640 2 cups, 1 bowl, 5 carrots, 1 dining table, Done. (0.011s)
    image 59/128 /content/coco128/images/train2017/000000000338.jpg: 352x640 2 persons, 1 oven, 1 sink, 2 refrigerators, 1 clock, Done. (0.010s)
    image 60/128 /content/coco128/images/train2017/000000000349.jpg: 480x640 2 potted plants, Done. (0.010s)
    image 61/128 /content/coco128/images/train2017/000000000357.jpg: 224x640 8 persons, 1 baseball bat, Done. (0.009s)
    image 62/128 /content/coco128/images/train2017/000000000359.jpg: 448x640 1 car, 2 traffic lights, Done. (0.010s)
    image 63/128 /content/coco128/images/train2017/000000000360.jpg: 480x640 1 person, Done. (0.011s)
    image 64/128 /content/coco128/images/train2017/000000000368.jpg: 640x480 9 persons, 2 sports balls, 2 kites, Done. (0.010s)
    image 65/128 /content/coco128/images/train2017/000000000370.jpg: 640x480 1 person, 1 broccoli, Done. (0.010s)
    image 66/128 /content/coco128/images/train2017/000000000382.jpg: 480x640 3 persons, 1 skis, Done. (0.010s)
    image 67/128 /content/coco128/images/train2017/000000000384.jpg: 640x448 1 couch, 6 books, 1 clock, Done. (0.011s)
    image 68/128 /content/coco128/images/train2017/000000000387.jpg: 480x640 Done. (0.010s)
    image 69/128 /content/coco128/images/train2017/000000000389.jpg: 480x640 12 persons, 2 ties, Done. (0.010s)
    image 70/128 /content/coco128/images/train2017/000000000394.jpg: 640x640 1 frisbee, Done. (0.013s)
    image 71/128 /content/coco128/images/train2017/000000000395.jpg: 608x640 8 persons, 1 cell phone, Done. (0.012s)
    image 72/128 /content/coco128/images/train2017/000000000397.jpg: 480x640 1 person, 1 pizza, 2 chairs, Done. (0.010s)
    image 73/128 /content/coco128/images/train2017/000000000400.jpg: 640x640 1 truck, 1 dog, Done. (0.012s)
    image 74/128 /content/coco128/images/train2017/000000000404.jpg: 640x448 1 boat, Done. (0.010s)
    image 75/128 /content/coco128/images/train2017/000000000415.jpg: 640x384 2 persons, 2 tennis rackets, Done. (0.010s)
    image 76/128 /content/coco128/images/train2017/000000000419.jpg: 480x640 2 persons, 2 cars, 2 tennis rackets, Done. (0.010s)
    image 77/128 /content/coco128/images/train2017/000000000428.jpg: 384x640 1 person, 1 cake, 1 remote, Done. (0.010s)
    image 78/128 /content/coco128/images/train2017/000000000431.jpg: 416x640 1 person, 1 sports ball, 1 tennis racket, Done. (0.010s)
    image 79/128 /content/coco128/images/train2017/000000000436.jpg: 640x448 1 person, 1 bottle, 1 donut, 1 clock, Done. (0.010s)
    image 80/128 /content/coco128/images/train2017/000000000438.jpg: 480x640 20 donuts, Done. (0.010s)
    image 81/128 /content/coco128/images/train2017/000000000443.jpg: 480x640 1 person, 1 remote, 1 cell phone, Done. (0.010s)
    image 82/128 /content/coco128/images/train2017/000000000446.jpg: 640x480 1 person, 1 sports ball, 4 potted plants, 1 remote, 1 cell phone, 3 books, Done. (0.010s)
    image 83/128 /content/coco128/images/train2017/000000000450.jpg: 480x640 1 bottle, 1 wine glass, 1 cup, 1 pizza, Done. (0.010s)
    image 84/128 /content/coco128/images/train2017/000000000459.jpg: 640x544 2 persons, 1 tie, 1 remote, Done. (0.011s)
    image 85/128 /content/coco128/images/train2017/000000000471.jpg: 448x640 1 bus, Done. (0.010s)
    image 86/128 /content/coco128/images/train2017/000000000472.jpg: 256x640 1 airplane, Done. (0.009s)
    image 87/128 /content/coco128/images/train2017/000000000474.jpg: 640x448 1 person, 1 baseball glove, Done. (0.010s)
    image 88/128 /content/coco128/images/train2017/000000000486.jpg: 448x640 1 bowl, 2 microwaves, 1 refrigerator, Done. (0.010s)
    image 89/128 /content/coco128/images/train2017/000000000488.jpg: 416x640 5 persons, 1 sports ball, 1 baseball bat, 2 baseball gloves, Done. (0.010s)
    image 90/128 /content/coco128/images/train2017/000000000490.jpg: 640x640 1 dog, 1 skateboard, Done. (0.013s)
    image 91/128 /content/coco128/images/train2017/000000000491.jpg: 416x640 1 bird, 1 dog, 1 teddy bear, Done. (0.010s)
    image 92/128 /content/coco128/images/train2017/000000000502.jpg: 448x640 1 bear, Done. (0.010s)
    image 93/128 /content/coco128/images/train2017/000000000508.jpg: 480x640 Done. (0.010s)
    image 94/128 /content/coco128/images/train2017/000000000510.jpg: 480x640 1 person, 1 bench, Done. (0.009s)
    image 95/128 /content/coco128/images/train2017/000000000514.jpg: 640x384 1 bed, Done. (0.011s)
    image 96/128 /content/coco128/images/train2017/000000000520.jpg: 480x640 2 persons, 8 birds, Done. (0.010s)
    image 97/128 /content/coco128/images/train2017/000000000529.jpg: 640x448 1 person, 1 motorcycle, Done. (0.010s)
    image 98/128 /content/coco128/images/train2017/000000000531.jpg: 480x640 9 persons, 3 tennis rackets, Done. (0.010s)
    image 99/128 /content/coco128/images/train2017/000000000532.jpg: 480x640 2 persons, 1 bus, 1 truck, 1 handbag, Done. (0.010s)
    image 100/128 /content/coco128/images/train2017/000000000536.jpg: 480x640 3 persons, 1 handbag, 1 clock, Done. (0.009s)
    image 101/128 /content/coco128/images/train2017/000000000540.jpg: 448x640 4 cars, 1 airplane, Done. (0.010s)
    image 102/128 /content/coco128/images/train2017/000000000542.jpg: 512x640 9 persons, Done. (0.010s)
    image 103/128 /content/coco128/images/train2017/000000000544.jpg: 448x640 10 persons, 1 sports ball, 1 baseball bat, 1 baseball glove, Done. (0.010s)
    image 104/128 /content/coco128/images/train2017/000000000560.jpg: 448x640 1 potted plant, 1 sink, 1 clock, Done. (0.009s)
    image 105/128 /content/coco128/images/train2017/000000000562.jpg: 640x448 3 toothbrushs, Done. (0.010s)
    image 106/128 /content/coco128/images/train2017/000000000564.jpg: 640x544 2 persons, 1 tennis racket, 30 chairs, Done. (0.011s)
    image 107/128 /content/coco128/images/train2017/000000000569.jpg: 480x640 1 person, 1 kite, Done. (0.010s)
    image 108/128 /content/coco128/images/train2017/000000000572.jpg: 640x448 2 persons, 1 suitcase, Done. (0.010s)
    image 109/128 /content/coco128/images/train2017/000000000575.jpg: 544x640 1 cat, Done. (0.011s)
    image 110/128 /content/coco128/images/train2017/000000000581.jpg: 640x608 1 dog, Done. (0.012s)
    image 111/128 /content/coco128/images/train2017/000000000584.jpg: 448x640 1 bowl, 1 broccoli, 7 carrots, 1 dining table, Done. (0.010s)
    image 112/128 /content/coco128/images/train2017/000000000589.jpg: 480x640 1 person, 1 frisbee, 1 sports ball, Done. (0.010s)
    image 113/128 /content/coco128/images/train2017/000000000590.jpg: 640x544 2 cups, 1 sink, Done. (0.011s)
    image 114/128 /content/coco128/images/train2017/000000000595.jpg: 480x640 1 tv, Done. (0.010s)
    image 115/128 /content/coco128/images/train2017/000000000597.jpg: 256x640 8 elephants, Done. (0.009s)
    image 116/128 /content/coco128/images/train2017/000000000599.jpg: 416x640 1 cat, 1 potted plant, 2 remotes, Done. (0.010s)
    image 117/128 /content/coco128/images/train2017/000000000605.jpg: 480x640 2 cups, 1 spoon, 1 cake, 1 dining table, Done. (0.010s)
    image 118/128 /content/coco128/images/train2017/000000000612.jpg: 480x640 2 vases, 6 teddy bears, Done. (0.010s)
    image 119/128 /content/coco128/images/train2017/000000000620.jpg: 640x480 1 pizza, 1 oven, Done. (0.010s)
    image 120/128 /content/coco128/images/train2017/000000000623.jpg: 640x480 1 person, 1 teddy bear, Done. (0.009s)
    image 121/128 /content/coco128/images/train2017/000000000625.jpg: 448x640 3 persons, 1 frisbee, Done. (0.010s)
    image 122/128 /content/coco128/images/train2017/000000000626.jpg: 480x640 2 clocks, Done. (0.010s)
    image 123/128 /content/coco128/images/train2017/000000000629.jpg: 448x640 1 motorcycle, Done. (0.010s)
    image 124/128 /content/coco128/images/train2017/000000000634.jpg: 640x448 6 persons, 1 skateboard, Done. (0.010s)
    image 125/128 /content/coco128/images/train2017/000000000636.jpg: 640x480 1 toilet, Done. (0.010s)
    image 126/128 /content/coco128/images/train2017/000000000641.jpg: 448x640 7 persons, 2 bicycles, 1 bus, 1 traffic light, Done. (0.010s)
    image 127/128 /content/coco128/images/train2017/000000000643.jpg: 480x640 1 bottle, 1 bowl, 1 tv, 1 book, Done. (0.010s)
    image 128/128 /content/coco128/images/train2017/000000000650.jpg: 448x640 1 cat, Done. (0.010s)
    Results saved to runs/detect/exp
    Done. (3.344s)
    


```python
from IPython.display import Image
Image('/content/runs/detect/exp/000000000009.jpg')
```




![jpeg](output_9_0.jpeg)




```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```

    Downloading: "https://github.com/ultralytics/yolov5/archive/master.zip" to /root/.cache/torch/hub/master.zip
    

    [31m[1mrequirements:[0m PyYAML>=5.3.1 not found and is required by YOLOv5, attempting auto-update...
    

    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
    

    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.7/dist-packages (5.4.1)
    
    [31m[1mrequirements:[0m 1 package updated per /root/.cache/torch/hub/ultralytics_yolov5_master/requirements.txt
    [31m[1mrequirements:[0m ⚠️ [1mRestart runtime or rerun command for updates to take effect[0m
    
    

      8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
      9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    Model Summary: 283 layers, 7276605 parameters, 7276605 gradients, 17.1 GFLOPS
    
    YOLOv5 🚀 2021-4-1 torch 1.8.1+cu101 CUDA:0 (Tesla T4, 15109.75MB)
    
    

    Adding autoShape... 
    


```python
model
```




    autoShape(
      (model): Model(
        (model): Sequential(
          (0): Focus(
            (conv): Conv(
              (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
          )
          (1): Conv(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (2): C3(
            (cv1): Conv(
              (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (3): Conv(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (4): C3(
            (cv1): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (5): Conv(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (6): C3(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
              (2): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (7): Conv(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (8): SPP(
            (cv1): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): ModuleList(
              (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
              (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
              (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
            )
          )
          (9): C3(
            (cv1): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (10): Conv(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (11): Upsample(scale_factor=2.0, mode=nearest)
          (12): Concat()
          (13): C3(
            (cv1): Conv(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (14): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (15): Upsample(scale_factor=2.0, mode=nearest)
          (16): Concat()
          (17): C3(
            (cv1): Conv(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (18): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (19): Concat()
          (20): C3(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (21): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (22): Concat()
          (23): C3(
            (cv1): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv2): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (cv3): Conv(
              (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
                (cv2): Conv(
                  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU()
                )
              )
            )
          )
          (24): Detect(
            (m): ModuleList(
              (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
              (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
              (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
      )
    )




```python
!wget https://wallpapersafari.com/download/l4Hekd/
```

    --2021-04-01 18:36:07--  https://wallpapersafari.com/download/l4Hekd/
    Resolving wallpapersafari.com (wallpapersafari.com)... 104.26.10.233, 172.67.74.226, 104.26.11.233, ...
    Connecting to wallpapersafari.com (wallpapersafari.com)|104.26.10.233|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [application/octet-stream]
    Saving to: ‘index.html’
    
    index.html              [  <=>               ]   1.25M  5.42MB/s    in 0.2s    
    
    2021-04-01 18:36:08 (5.42 MB/s) - ‘index.html’ saved [1308818]
    
    


```python
!ls
```

    coco128					     runs	  yolov5s.pt
    index.html				     sample_data  yolov5x.pt
    itl.cat_free-hd-nature-wallpaper_931441.png  yolov5
    


```python
import cv2 as cv2

img = cv2.imread("/content/Hd Bugatti Supercar Wallpaper 2   SA Wallpapers.jpg")
cv2_imshow(img)
```


![png](output_14_0.png)



```python
img_8000 = cv2.resize(img, (8000, 6000))
cv2_imshow(img_8000)
```


![png](output_15_0.png)



```python
cv2.imwrite('/content/Hd-Bugatti_8000.jpg', img_8000)
img_8000 = 0
```


```python
!mkdir test
```


```python
!cp /content/Hd-Bugatti_8000.jpg /content/test/
```


```python
!cp "/content/Hd Bugatti Supercar Wallpaper 2   SA Wallpapers.jpg" /content/test/
```


```python
!python /content/yolov5/detect.py --source /content/test --weights yolov5s.pt --conf 0.25
```

    Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='/content/test', update=False, view_img=False, weights=['yolov5s.pt'])
    [31m[1mrequirements:[0m /content/requirements.txt not found, check failed.
    YOLOv5 🚀 v4.0-180-g1148e2e torch 1.8.1+cu101 CUDA:0 (Tesla T4, 15109.75MB)
    
    Fusing layers... 
    Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
    image 1/2 /content/test/Hd Bugatti Supercar Wallpaper 2   SA Wallpapers.jpg: 384x640 1 car, Done. (0.011s)
    image 2/2 /content/test/Hd-Bugatti_8000.jpg: 480x640 1 car, Done. (0.011s)
    Results saved to runs/detect/exp2
    Done. (1.723s)
    

# Result of Object detection of "1240 X ___" image


```python
img = cv2.imread("/content/runs/detect/exp2/Hd Bugatti Supercar Wallpaper 2   SA Wallpapers.jpg")
cv2_imshow(img)
img = 0
```


![png](output_22_0.png)


# Result of Object Detection of "8000 x ____" image

the image shown is resize to (2000, 1000) in order to reduce the display size
that is the reason the rectangle boundary are thin


```python
img_8000 = cv2.imread("/content/runs/detect/exp2/Hd-Bugatti_8000.jpg")
img_2000 = cv2.resize(img_8000, (2000, 1000))
cv2_imshow(img_2000)
```


![png](output_24_0.png)



```python

```


