# Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET



The focus of this term project is to tackle image classification problems using a variety of methods on two different datasets and compare the performance of each method using confusion matrix, accuracy results and Receiver Operation Characteristic (ROC).
The first dataset was chosen as CIFAR-10 which has 10 classes, and 50000 training and 10000 test images. The second dataset was chosen as MNISET which has around 3000 training and 1000 test images with 27 different SET card classes. Each dataset will be classified using Neural Network, K-Nearest-Neighbors, and Logistic Regression and performance of each model with each other as well as on different datasets will be discussed in this report.




![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/867f939d-c190-48f8-946b-a3746748bebb)


CIFAR-10 Dataset
This dataset consists of 10 classes which are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class has 6000 32×32 RGB images; however, they were converted to grayscale using luminosity grayscale conversion in this project to reduce complexity of the problem. It is split into 40000 training, 1000 validation and 10000 test images in the project. This dataset is quite complex such that average human can classify it with %95 accuracy and has been used as a benchmark for many computers vision and machine learning studies. Thus, it will be a challenging dataset to solve with the proposed methods and a good way to assess robustness of our methods. 
MNISET Dataset
This dataset generated while design of SET Finder app [2] and has nearly 4000 28×28 greyscale images with 27 different SET card samples. Like CIFAR-10 images converted grayscale. This dataset split into 2429 training,607 validation and 922 test images. Each class has around 110 images with different lighting, and orientation which makes this dataset more complex than MNIST dataset but not as complex as CIFAR-10.  The distribution of each class as well as sample for each class van be seen in following figure.


![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/2516feb5-6280-409e-a3bf-81ae3610ee4a)


![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/8c1f86ad-3e0a-402b-98d5-7f9811cfa34d)



![image](https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/111acdd7-03f1-4523-af64-8e17f26793bf)
