# 机器学习工程师纳米学位
## 猫狗大战项目报告


**注意**

**第一步，**下载VGGNet:

首先要下载VGGNet相关文件，这是一个已训练过的VGG网络，来自 https://github.com/machrisaa/tensorflow-vgg 的一个预训练网络。确保将此目录克隆到你猫狗大战下的目录。

当然你也可以使用githud : git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg

[image4]:./image/4.jpg "image1"
![alt text][image4] 


**第二步，**下载VGGNet参数：

紧接着要下载VGGNet的参数,大小为500多M，下载完后记得放在tensorflow_vgg目录下，下载地址为：https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy

[image3]:./image/3.jpg "image1"
![alt text][image3] 

**第三步，**下载项目数据：

直接在Kaggle猫狗竞赛官方网站下载：

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

[image2]:./image/2.jpg "image1"
![alt text][image2] 


**第四步：**获取VGGNet特征提取器

完成以上三步后就可以训练VGGNet，在GPU下运行，整个过程大概20-30分钟，完成训练后就能够提取特征文件，然后保存在对应项目目录下，两个文件分别为：

特征文件：codes，标签文件：labels，两个总大小为390M。

[image1]:./image/1.jpg "image1"
![alt text][image1] 

**第五步：**训练微调后的VGGNet网络

当你完成对VGGNet网络的全连接层调整后，就能直接拿特征文件来训练全连接层了，训练需几十秒，训练结果如下：

打印文本：

[image5]:./image/5.jpg "image1"
![alt text][image5] 

曲线图：


[image6]:./image/6.jpg "image1"
![alt text][image6] 

详细可以查看附带的report.html文件。