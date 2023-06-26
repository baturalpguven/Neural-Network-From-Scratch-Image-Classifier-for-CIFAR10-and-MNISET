# Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET



The focus of this project is to tackle image classification problems using a simple fully connected neural network on two different datasets and compare the performance of each method using a confusion matrix, accuracy results, and Receiver Operation Characteristic (ROC).
The first dataset was chosen as CIFAR-10 which has 10 classes, 50000 training, and 10000 test images. The second dataset was chosen as <a href="https://github.com/wouterkool/MNISET"> MNISET </a> which has around 3000 training and 1000 test images with 27 different SET card classes. 

## Architecture of the Proposed Network

<p align="center">
<img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/867f939d-c190-48f8-946b-a3746748bebb" align = "center" width="50%" height="50%">
</p>


CIFAR-10 Dataset
This dataset consists of 10 classes which are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class has 6000 32×32 RGB images; however, they were converted to grayscale using luminosity grayscale conversion in this project to reduce the complexity of the problem. It is split into 40000 training, 1000 validation, and 10000 test images in the project. This dataset is quite complex such that the average human can classify it with %95 accuracy and has been used as a benchmark for many computer vision and machine learning studies. Thus, it will be a challenging dataset to solve with the proposed methods and a good way to assess the robustness of our methods. 
MNISET Dataset
This dataset was generated while designing of SET Finder app and has nearly 4000 28×28 greyscale images with 27 different SET card samples. Like CIFAR-10 images converted to grayscale. This dataset is split into 2429 training,607 validation, and 922 test images. Each class has around 110 images with different lighting, and orientation which makes this dataset more complex than the MNIST dataset but not as complex as CIFAR-10.  The distribution of each class, as well as samples for each class, can be seen in the following figure.
<p align="center">
<img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/2516feb5-6280-409e-a3bf-81ae3610ee4a" align = "center" >
</p>

## PCA Analysis
![Dataset mniset PVE Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/f9bbf7e0-c4e4-47a1-ba84-553ad330bc2a)
![Dataset cifar10 PVE Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/963524b3-80d3-4619-bb0c-10acdfdb058b)


## Results
![Dataset cifar10 Acc Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/008e8db4-9786-4448-99ce-95343462aea8)

![Dataset cifar10 Confusion Matrix](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/26530b8e-570f-43fe-a1b7-304e2d1782fe)

![Dataset cifar10 Loss Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/86defc23-4db8-4584-be83-62ab30ce4581)


![Dataset cifar10 ROC Curve](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/fd5a3f3b-41ec-4f6c-8e56-aafa96f403bd)


![Dataset cifar10 Visual Samples](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/319da9eb-68f9-4373-8552-20bfa3a92747)


![Dataset mniset Acc Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/1cecca61-98c9-4c0f-874d-fbd4d08d310c)



![Dataset mniset Confusion Matrix](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/83da1f52-6c9c-4530-a909-a9f868ff24df)



![Dataset mniset Loss Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/99cba36b-c708-4d05-82a3-f71b327154f2)





## Running the Code

## Referances

1. <a href="https://nthu-datalab.github.io/ml/index.html"> CS565600 Deep Learning, National Tsing Hua University </a>
2. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/"> Building a Neural Network from Scratch: Part 1 </a>
3. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/"> Building a Neural Network from Scratch: Part 2 </a>
4. <a href="https://developer.ibm.com/technologies/artificial-intelligence/articles/neural-networks-from-scratch/"> Neural networks from scratch, IBM Developer</a>
5. <a href="https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/"> The Softmax Function Derivative (Part 1) </a>
6. <a href="https://github.com/lionelmessi6410/Neural-Networks-from-Scratch"> Neural-Networks-from-Scratch </a>







![Dataset mniset ROC Curve](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/746542d5-49f7-4350-98da-a380c3497b6c)



![Dataset mniset Visual Samples](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/c379252f-56b3-487f-b0ce-b9ec936763c7)
