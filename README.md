# huaweicloud_garbage_classify_competiton
introduction
------  
  This project is a conclusion about the experience in the competition["Huawei cloud artificial intelligence competition-Garbage sorting challenge cup](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction).This issue contains 40 kinds of [garbage images](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify.zip) gathered by daily life. Each pair of data includes a garbage image and its label file in TXT format within a line of image name and its corresponding digit label, such as 'image_0.jpg,0'(name,label).The index of data is not continous and the total number of images is about 14802.

Work
------
  This issue seems like a simple image classification problem. Mostly, the scheme of solving this kind of problems is split into the following steps.(Cite:[Scheme for Kaggle seedling classification contest](https://baijiahao.baidu.com/s?id=1604481732386439544&wfr=spider&for=pc))
### Data Statistic and Analysis
Firstly, we need to do some data statistic and analysis before we build our model.I count the total image number for each class,and get the following data distribution histogram.
![](https://github.com/lpf9562/huaweicloud_garbage_classify_competiton/master/data_distribution.png)
