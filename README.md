# DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal

[Cong Wang](https://supercong94.wixsite.com/supercong94)\*, Xiaoying Xing\*, [Yutong Wu](https://github.com/Ohraincu), [Zhixun Su](http://faculty.dlut.edu.cn/ZhixunSu/zh_CN/index/759047/list/index.htm) , Junyang Chen †

<\* Both authors contributed equally to this research. † Corresponding author.>

This work has been accepted by ACM'MM 2020. 

## Abstract
Rain removal is an important but challenging computer vision task as rain streaks can severely degrade the visibility of images that may make other visions or multimedia tasks fail to work. Previous works mainly focused on feature extraction and processing or neural network structure, while the current rain removal methods can already achieve remarkable results, training based on single network structure without considering the cross-scale relationship may cause information drop-out. In this paper, we explore the cross-scale manner between networks and inner-scale fusion operation to solve the image rain removal task. Specifically, to learn features with different scales, we propose a multi-sub-networks structure, where these sub-networks are fused via a cross-scale manner by Gate Recurrent Unit to inner-learn and make full use of information at different scales in these sub-networks. Further, we design an inner-scale connection block to utilize the multi-scale information and features fusion way between different scales to improve rain representation ability and we introduce the dense block with skip connection to inner-connect these blocks. Experimental results on both synthetic and real-world datasets have demonstrated the superiority of our proposed method, which outperforms over the state-of-the-art methods.


## Requirements
- CUDA 9.0
- Python 3.6 (or later)
- Pytorch 1.1.0
- Torchvision 0.3.0
- OpenCV

## Dataset
Please download the following datasets:

* Rain100L [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain100H [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain1200 [[dataset](https://github.com/hezhangsprinter/DID-MDN)]
* Real-world images [[dataset](https://pan.baidu.com/s/1gpuB6NUHPnQEtgRn4evr5A)](password:vvuk)

## Setup
Please download this project through 'git' command.
```
$ git clone https://github.com/Ohraincu/DCSFN.git
$ cd config
```  
Thanks to [the code by Li et al.](https://xialipku.github.io/RESCAN/), our code is also adapted based on this.

## Training
After you download the above datasets, you can perform the following operations to train:
```
$ python train.py
```  
You can pause or start the training at any time because we can save the pre-trained models in due course.

## Testing
### Pre-trained Models
[[BaiduYun]](https://pan.baidu.com/s/19GO38UnOlMKcDToGJyNwvw)(pw:c3ri)

### Quantitative and Qualitative Results
After running eval.py, you can get the corresponding numerical results (PSNR and SSIM):
```
$ python eval.py
``` 
If the visual results on datasets need to be observed, the show.py can be run:
```
$ python show.py
``` 
All the pre-trained model in each case is placed in the corresponding 'model' folder, and the 'latest_net' model is directly referenced by default. 


## Citation
```
@inproceedings{acmmm20_jdnet,
	author    = {Cong Wang and Xiaoying Xing and Yutong Wu and Zhixun Su and Junyang Chen},
	title     = {DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal},
	booktitle = {ACM International Conference on Multimedia},
	year      = {2020},
}
```

## Contact

If you are interested in our work or have any questions, please directly contact my github.
Email: supercong94@gmail.com / ytongwu@mail.dlut.edu.cn
