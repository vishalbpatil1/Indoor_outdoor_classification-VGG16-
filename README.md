# Indoor_outdoor_classification-VGG16-

### VGG16 implementation
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014.
Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.
I am going to implement full VGG16 from scratch in Keras. This implement will be done on Indoor vs Outdoor dataset.
[link to keras doc](https://keras.io/api/applications/vgg/)

### Reasult outdoor image
![outdoor](https://github.com/vishalbpatil1/Indoor_outdoor_classification-VGG16-/blob/main/Screenshot%20(152).png)

### Result indoor image
![indoor](https://github.com/vishalbpatil1/Indoor_outdoor_classification-VGG16-/blob/main/Screenshot%20(154).png)

### classification report
![report](https://github.com/vishalbpatil1/Indoor_outdoor_classification-VGG16-/blob/main/classification%20report.png)


