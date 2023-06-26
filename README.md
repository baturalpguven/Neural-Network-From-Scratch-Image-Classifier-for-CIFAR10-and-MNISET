# Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET


The focus of this project is to tackle image classification problems using a simple fully connected neural network on two different datasets and compare the performance of each method using a confusion matrix, accuracy results, and Receiver Operation Characteristic (ROC).
The first dataset was chosen as CIFAR-10 which has 10 classes, 50000 training, and 10000 test images. The second dataset was chosen as <a href="https://github.com/wouterkool/MNISET"> MNISET </a> which has around 3000 training and 1000 test images with 27 different SET card classes. 

## Architecture of the Proposed Network

This method is popular due to its robustness and efficiency in various problems, including simple image classification like MNIST. However, convolution-based approaches are more popular for image classification as they preserve structural details and meaningful features. The method multiplies features by adjustable weights, applies non-linear activation functions like ReLU and Softmax, and updates weights using gradient computation of a loss function. Lecture notes and [3,4,5,6,7] were used as main sources in implementation. Regularization and PCA analysis were applied to handle complexity due to the fully connected structure's high dimensional features. ADAM optimizer and mini-batch training were implemented to improve performance. Empirical results showed that 2 hidden layers with sizes of 64 and 32, l_2regularization, PCA, 64 batch size, and ADAM optimizer significantly improved the results. The figure illustrates the network's number of hidden layers and layer sizes:

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

First, let’s start with implementation of PCA, this methos is one of the most common feature reduction methods that preserves the variance of the dataset choosing most relevant features via assessing their eigenvectors as much as possible. Due to this reason, it was an important tool to use in our work. Mathematical model of PCA can be summarized as 
$$\Sigma = X^T X$$
$$\Sigma u = \lambda u$$
$$(\Sigma - \lambda_m I) u_m = 0$$



Where X is data matrix,Σ is covariance matrix,u is eigenvectors and λ is eigenvalues. Subscript i indicates number of eigenvalues that is selected. Finally, x_new is the representation of the dataset on lower dimensions.  To assess how much of the variance in the original dataset is used Proportion of Variance Explained (PVE) equation calculated by diving variance of mth principal component to total variance, which can be mathematically explained as
$$PVE(m) = \frac{\sum_{i=1}^n (x_i^T u_m)^2}{\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}$$


For our case PVE showed that


<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/f9bbf7e0-c4e4-47a1-ba84-553ad330bc2a" alt="Dataset mniset PVE Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/963524b3-80d3-4619-bb0c-10acdfdb058b" alt="Dataset cifar10 PVE Graph">
    </td>
  </tr>
</table>



For CIFAR-10 the number of features from these graph chosen as 250, and for MNISET 150.

## ADAM

Lastly, lets discuss ADAM implementation in the project for utilizing momentum which helps SGD to escape local minima via adding a fraction of previous gradient and RMSProp which solves vanishing gradient problem of the neural networks via taking moving averages of the squared gradients and generalizes better than Stochastic gradient descent (SGD). In other words, ADAM uses the second moment of to update bias terms as well as momentum. The following equation summarizes ADAM.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla w_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla w_t)^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_{t+1}= w_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}}\hat{m}_t$$


All these improvements applied to the model to achieve stability, faster convergences with faster time.


# Results

The proposed method got an accuracy of %67 on MNISET dataset in 2400 epoch, and confusion matrix as well as ROC graphs indicates that Neural network can solve MNISET dataset to a certain extent. Implementing regularization, ADAM and PCA significantly improved model via reducing its complexity and forcing network to get into a better point on bias variance trade off.
Resultant graph for MNISET can be seen in the following graphs:

## Results CIFAR-10


<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/008e8db4-9786-4448-99ce-95343462aea8" alt="Dataset cifar10 Acc Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/26530b8e-570f-43fe-a1b7-304e2d1782fe" alt="Dataset cifar10 Confusion Matrix">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/86defc23-4db8-4584-be83-62ab30ce4581" alt="Dataset cifar10 Loss Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/fd5a3f3b-41ec-4f6c-8e56-aafa96f403bd" alt="Dataset cifar10 ROC Curve">
    </td>
  </tr>
</table>

## Results MNISET

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/1cecca61-98c9-4c0f-874d-fbd4d08d310c" alt="Dataset mniset Acc Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/83da1f52-6c9c-4530-a909-a9f868ff24df" alt="Dataset mniset Confusion Matrix">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/99cba36b-c708-4d05-82a3-f71b327154f2" alt="Dataset mniset Loss Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/746542d5-49f7-4350-98da-a380c3497b6c" alt="Dataset mniset ROC Curve">
    </td>
  </tr>
</table>


For the CIFAR-10 dataset it can be said that model overfitted and could not achieved higher accuracy higher than %44 accuracy in 2400 epoch. This is due to complex data inside of CIFAR-10 and indicates that PCA, and regularization is not enough to reduce the complexity of the model and select features. Convolutional based approach is necessary and will be implemented in the final report.

## Visual Samples

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/319da9eb-68f9-4373-8552-20bfa3a92747" alt="Dataset cifar10 Visual Samples" >
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/c379252f-56b3-487f-b0ce-b9ec936763c7" alt="Dataset mniset Visual Samples" >
    </td>
  </tr>
</table>


## Running the Code
```
To run the code simply run train.py but you can adjust train and eval modes with commands.
```

## Referances

1. <a href="https://nthu-datalab.github.io/ml/index.html"> CS565600 Deep Learning, National Tsing Hua University </a>
2. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/"> Building a Neural Network from Scratch: Part 1 </a>
3. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/"> Building a Neural Network from Scratch: Part 2 </a>
4. <a href="https://developer.ibm.com/technologies/artificial-intelligence/articles/neural-networks-from-scratch/"> Neural networks from scratch, IBM Developer</a>
5. <a href="https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/"> The Softmax Function Derivative (Part 1) </a>
6. <a href="https://github.com/lionelmessi6410/Neural-Networks-from-Scratch"> Neural-Networks-from-Scratch </a>








