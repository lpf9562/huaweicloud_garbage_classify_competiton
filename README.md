# huaweicloud_garbage_classify_competiton
Introduction
------  
  This project is a conclusion about the experience in the competition["Huawei cloud artificial intelligence competition-Garbage sorting challenge cup](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction).This issue contains 40 kinds of [garbage images](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify.zip) gathered by daily life. Each pair of data includes a garbage image and its label file in TXT format within a line of image name and its corresponding digit label, such as 'image_0.jpg,0'(name,label).The index of data is not continous and the total number of images is about 14802.

Work
------
  This issue seems like a simple image classification problem. Mostly, the scheme of solving this kind of problems is split into the following steps.(Cite:[Scheme for Kaggle seedling classification contest](https://baijiahao.baidu.com/s?id=1604481732386439544&wfr=spider&for=pc))
### Data Statistic and Analysis
Firstly, we need to do some data statistic and analysis before we build our model.I count the total image number for each class,and get the following data distribution histogram.
![](https://github.com/lpf9562/huaweicloud_garbage_classify_competiton/blob/master/data_distribution.png)
As we can see in the picture, the number of each class is not balanced.The max is 736 and the min is 85. So we need to make the data balanced. We can use minus data up-sampling to make it.The same as the next part-Data augmentation, we can use some python external lib such as keras-datagenerator, imgaug, or some manual functions by yourself. Besides, [t-distributed Stochastic Neighbor Embedding(t-SNE)](http://lvdmaaten.github.io/tsne/) is also a good method for data analysis. But some problems raised when I use this method, the cpu memory blowed up. Maybe the way I used was wrong.
I wanted to make the data balanced, so I up-sampled minus image data with keras-ImageDataGenerator. The number of each class was increased to 10,000. But the result accuracy was not improved, reduced instead. So it illustrates that large amount of repeating image data can't help. 
### Data Augmentation
This a significant part for data processing science. There are three kinds of data-aug method I used during the proceed.The first is [keras-ImageDataGenerator](https://keras.io/zh/preprocessing/image/).
```python
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1. / 255,#image channels normalization with divided by 255
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,#crop
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
val_generator = datagen.flow(batch_x,batch_y,
                             target_size=(img_size, img_size),
                             save_to_dir='E:\garbage_classify\garbage_classify\\test',#augmented image address
                             save_prefix='image',
                             save_format='jpg',
                             batch_size=40,
                             shuffle=False)
#or we can augment data from its directory
val_generator = datagen.flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False)
#these two function will output a generator, so that we can us this for
model.fit_generator(val_generator,
                            steps_per_epoch = 100,
                            epochs= 20,
                            validation_data = validation_generator, 
                            validation_steps= 50)
#or we can use following function to applies a transformation to an image according to given parameters, this can be applied to sequence data input.
img=datagen.apply_transform(x,parameters)
```
The second method for image-aug is imgaug, some blogger has summrized for a detailed reference [Imgaug data enhancement library - study notes](https://blog.csdn.net/qq_38451119/article/details/82428612):
```python
from imgaug import augmenters as iaa
seq = iaa.Sequential([#iaa.OneOf,iaa.SomeOf
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
```
The last method for image-aug is self-manual augmentation method, the following is my method including rotation, flip, Gaussian blur, gamma_transform, add_noise or something else:
```python
    def gamma_transform(self,img, gamma=1):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)
    def random_gamma_transform(self,img, gamma_vari=2):
        log_gamma_vari = np.log(gamma_vari)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        gamma = np.exp(alpha)
        return self.gamma_transform(img, gamma)
    def rotate(self,img,angle):
        size=224
        M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        img = cv2.warpAffine(img, M_rotate, (size, size))
        return img
    def blur(self,img):
        img = cv2.blur(img, (3, 3))
        return img
    def add_noise(self,img):
        size=100
        for i in range(size):  # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img
    def data_augment(self,img):
        if np.random.random() < 0.25:
            img = self.rotate(img,90)
        if np.random.random() < 0.25:
            img = self.rotate(img, 180)
        if np.random.random() < 0.25:
            img = self.rotate(img, 270)
        if np.random.random() < 0.25:
            img = cv2.flip(img, 1)  # flipcode > 0：沿y轴翻转
        if np.random.random() < 0.25:
            img = self.random_gamma_transform(img, 1)
        if np.random.random() < 0.25:
            img = self.blur(img)
        if np.random.random() < 0.25:
            img = cv2.bilateralFilter(img, 9, 75, 75)
        if np.random.random() < 0.25:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if np.random.random() < 0.2:
            img = self.add_noise(img)
        return img
```
By applying data augmentation the test accuracy will be improved by at least 3%.
### Build Model
Official Huawei Cloud provided a baseline built by resnet-50.Run the baseline we can get a score with 66%, which is not enough at all.So I decided to change the model.But simply using one model may not learn whole image feature, I used the model stacking technology. [Keras application](https://keras.io/applications/) has provided some state of the art model and their trained weighted on imageNet, which is convenient and useful.
.<div align=center><img src="https://github.com/lpf9562/huaweicloud_garbage_classify_competiton/blob/master/keras-application.png"  /></div>
We can concatenate convolutional outputs of models using
```python
input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=True,
                                      weights='imagenet')
inp=Input(input_shape) 
base_model1 = InceptionV3(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
base_model2 = Xception(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
base_model3=ResNet50(
                          input_tensor=inp,
                          weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
for layer in base_model1.layers:
        layer.trainable = True
for layer in base_model2.layers:
        layer.trainable = True
for layer in base_model3.layers:
        layer.trainable =True
inception = base_model1.output
xception=base_model2.output
densenet=base_model3.output
top1_model=GlobalMaxPooling2D(data_format='channels_last')(inception)
top2_model = GlobalMaxPooling2D(data_format='channels_last')(xception)
top3_model = GlobalMaxPooling2D(data_format='channels_last')(densenet)
top1_model = Flatten()(top1_model)
top2_model = Flatten()(top2_model)
top3_model = Flatten()(top3_model)
t=concatenate([top1_model,top2_model,top3_model],axis=1)
predictions = Dense(FLAGS.num_classes, activation='softmax')(t)
model = Model(inputs=inp, outputs=predictions)
model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
```
But this way will lead to a large model size, which is not ideal.So we can also average several softmax outputs to obtain a ave prediction. By communicating with some competition participators, I got that [Efficientnet](https://arxiv.org/abs/1905.11946) was a pretty state of the art model with good performance. But what a pitty, the submission deadline has come.The Efficientnet integrats several state of the art tricks, for example:width scaling, depth scaling, resolution scaling to learn full-scale features.
![](https://github.com/lpf9562/huaweicloud_garbage_classify_competiton/blob/master/efficientnet.png)
The baseline objective function is 'binary_crossentropy' which fits binary classification. For multi-classification we can use 'categorical_crossentropy'.[Keras_losses](https://keras.io/losses/)
The keras provides several state of the art [optimizer](https://keras.io/zh/optimizers/), we can make a choice between Adam and Nadam.
```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)#parameters all come from its original paper
```
Conclusion and Future Work
---------
This is my first time to attend this kind of AI image processing competition. Because lacking of experience and corresponding knowledge,
the final best score was 87% in rank 156. 
![](https://github.com/lpf9562/huaweicloud_garbage_classify_competiton/blob/master/score.png)
There is still very much to learn, write this repository to record what I obtained during this proceeding. I feel appreciate for it much. 
