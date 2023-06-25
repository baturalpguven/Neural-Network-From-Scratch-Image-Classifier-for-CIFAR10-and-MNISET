# Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET



The focus of this term project is to tackle image classification problems using a variety of methods on two different datasets and compare the performance of each method using confusion matrix, accuracy results and Receiver Operation Characteristic (ROC).
The first dataset was chosen as CIFAR-10 which has 10 classes, and 50000 training and 10000 test images. The second dataset was chosen as MNISET which has around 3000 training and 1000 test images with 27 different SET card classes. Each dataset will be classified using Neural Network, K-Nearest-Neighbors, and Logistic Regression and performance of each model with each other as well as on different datasets will be discussed in this report.




![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/867f939d-c190-48f8-946b-a3746748bebb)


CIFAR-10 Dataset
This dataset consists of 10 classes which are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class has 6000 32×32 RGB images; however, they were converted to grayscale using luminosity grayscale conversion in this project to reduce complexity of the problem. It is split into 40000 training, 1000 validation and 10000 test images in the project. This dataset is quite complex such that average human can classify it with %95 accuracy and has been used as a benchmark for many computers vision and machine learning studies. Thus, it will be a challenging dataset to solve with the proposed methods and a good way to assess robustness of our methods. 
MNISET Dataset
This dataset generated while design of SET Finder app [2] and has nearly 4000 28×28 greyscale images with 27 different SET card samples. Like CIFAR-10 images converted grayscale. This dataset split into 2429 training,607 validation and 922 test images. Each class has around 110 images with different lighting, and orientation which makes this dataset more complex than MNIST dataset but not as complex as CIFAR-10.  The distribution of each class as well as sample for each class van be seen in following figure.


![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/2516feb5-6280-409e-a3bf-81ae3610ee4a)


![Dataset cifar10 Acc Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/008e8db4-9786-4448-99ce-95343462aea8)

![Dataset cifar10 Confusion Matrix](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/26530b8e-570f-43fe-a1b7-304e2d1782fe)

![Dataset cifar10 Loss Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/86defc23-4db8-4584-be83-62ab30ce4581)


![Dataset cifar10 PVE Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/963524b3-80d3-4619-bb0c-10acdfdb058b)



![Dataset cifar10 ROC Curve](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/fd5a3f3b-41ec-4f6c-8e56-aafa96f403bd)


![Dataset cifar10 Visual Samples](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/319da9eb-68f9-4373-8552-20bfa3a92747)


![Dataset mniset Acc Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/1cecca61-98c9-4c0f-874d-fbd4d08d310c)



![Dataset mniset Confusion Matrix](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/83da1f52-6c9c-4530-a909-a9f868ff24df)



![Dataset mniset Loss Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/99cba36b-c708-4d05-82a3-f71b327154f2)



![Dataset mniset PVE Graph](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/f9bbf7e0-c4e4-47a1-ba84-553ad330bc2a)



![Dataset mniset ROC Curve](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/746542d5-49f7-4350-98da-a380c3497b6c)



![Dataset mniset Visual Samples](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/c379252f-56b3-487f-b0ce-b9ec936763c7)
