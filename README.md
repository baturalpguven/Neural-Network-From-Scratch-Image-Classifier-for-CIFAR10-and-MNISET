# Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET


The focus of this project is to tackle image classification problems using a simple fully connected neural network on two different datasets and compare the performance of each method using a confusion matrix, accuracy results, and Receiver Operation Characteristic (ROC).
The first dataset was chosen as CIFAR-10 which has 10 classes, 50000 training, and 10000 test images. The second dataset was chosen as <a href="https://github.com/wouterkool/MNISET"> MNISET </a> which has around 3000 training and 1000 test images with 27 different SET card classes. 

## Architecture of the Proposed Network

This method is popular due to its robustness and efficiency in various problems, including simple image classification like MNIST. However, convolution-based approaches are more popular for image classification as they preserve structural details and meaningful features which is why PCA is applied to improve accuracy. The method multiplies features by adjustable weights applies non-linear activation functions like ReLU and Softmax, and updates weights using gradient computation of a loss function. Regularization and PCA analysis were applied to handle complexity due to the fully connected structure's high dimensional features. ADAM optimizer and mini-batch training were implemented to improve performance. Empirical results showed that 2 hidden layers with sizes of 64 and 32, $l_2$ regularization, PCA, 64 batch size, and ADAM optimizer significantly improved the results. The figure illustrates the network's number of hidden layers and layer sizes:



<p align="center">
<img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/b02c5cb1-d3ff-4ccd-a017-d6c6f414cd17" align = "center" width="50%" height="50%">
</p>


CIFAR-10 Dataset
This dataset consists of 10 classes which are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class has 6000 32×32 RGB images; however, they were converted to grayscale using luminosity grayscale conversion in this project to reduce the complexity of the problem. It is split into 40000 training, 1000 validation, and 10000 test images in the project. This dataset is quite complex such that the average human can classify it with %95 accuracy and has been used as a benchmark for many computer vision and machine learning studies. Thus, it will be a challenging dataset to solve with the proposed methods and a good way to assess the robustness of our methods. 
<a href="https://github.com/wouterkool/MNISET"> MNISET </a> Dataset
This dataset was generated while designing of SET Finder app and has nearly 4000 28×28 greyscale images with 27 different SET card samples. Like CIFAR-10 images converted to grayscale. This dataset is split into 2429 training,607 validation, and 922 test images. Each class has around 110 images with different lighting, and orientation which makes this dataset more complex than the MNIST dataset but not as complex as CIFAR-10.  The distribution of each class, as well as samples for each class, can be seen in the following figure.
<p align="center">
<img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/cfc896b9-9ae0-4dd6-b2a6-565359652b3f" align = "center">
</p>

## PCA Analysis

First, let’s start with the implementation of PCA, this method is one of the most common feature reduction methods that preserve the variance of the dataset by choosing the most relevant features via assessing their eigenvectors as much as possible. Due to this reason, it was an important tool to use in our work. The mathematical model of PCA can be summarized as 
$$\Sigma = X^T X$$
$$\Sigma u = \lambda u$$
$$(\Sigma - \lambda_m I) u_m = 0$$



Where X is the data matrix, Σ is the covariance matrix,u is the eigenvectors and λ is the eigenvalues. The subscript i indicates a number of eigenvalues that are selected. Finally, $x_new$ is the representation of the dataset on lower dimensions.  To assess how much of the variance in the original dataset is used Proportion of Variance Explained (PVE) equation is calculated by diving variance of the mth principal component to the total variance, which can be mathematically explained as
$$PVE(m) = \frac{\displaystyle\sum_{i=1}^n (x_i u_m)^2}{\displaystyle\sum_{j=1}^p \displaystyle\sum_{i=1}^n x_{ij}^2}$$


In our case, PVE showed that


<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/5ee273a0-d654-4636-ae9a-9ec108b22448" alt="Dataset mniset PVE Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/b99610ac-8869-4ddd-b7e9-3ad87e439d8d" alt="Dataset cifar10 PVE Graph">
    </td>
  </tr>
</table>



For CIFAR-10 the number of features from this graph chosen was 250, and for <a href="https://github.com/wouterkool/MNISET"> MNISET </a> 150.

## ADAM

Lastly, let's discuss ADAM implementation in the project for utilizing momentum which helps SGD to escape local minima via adding a fraction of the previous gradient, and RMSProp which solves the vanishing gradient problem of the neural networks via taking moving averages of the squared gradients and generalizes better than Stochastic gradient descent (SGD). In other words, ADAM uses the second moment to update bias terms as well as momentum. The following equation summarizes ADAM.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla w_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla w_t)^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$w_{t+1}= w_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}}\hat{m}_t$$


All these improvements are applied to the model to achieve stability, and faster convergences in a shorter time.


# Results

The proposed method got an accuracy of %67 on the <a href="https://github.com/wouterkool/MNISET"> MNISET </a> dataset in 2400 epochs, and the confusion matrix, as well as ROC graphs, indicates that the Neural network can solve the <a href="https://github.com/wouterkool/MNISET"> MNISET </a> dataset to a certain extent. Implementing regularization, ADAM and PCA significantly improved the model by reducing its complexity and forcing the network to get to a better point on bias-variance trade-off.
Resultant graph for <a href="https://github.com/wouterkool/MNISET"> MNISET </a> can be seen in the following graphs:

## Results CIFAR-10




<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/16e0162d-3f19-4bbd-8f77-2f92d65fbf8d" alt="Dataset cifar10 Acc Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/efc7aaec-37c5-4f77-bd48-63e58a3b592b" alt="Dataset cifar10 Confusion Matrix">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/876cae20-b6d5-421c-8ad2-a61bb479e782" alt="Dataset cifar10 Loss Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/c27d648f-418e-4ab5-99b4-192fa22ac22f" alt="Dataset cifar10 ROC Curve">
    </td>
  </tr>
</table>

## Results <a href="https://github.com/wouterkool/MNISET"> MNISET </a>

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/4134a8f4-d364-46bd-bf70-86b105d1a87c" alt="Dataset mniset Acc Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/020b289d-7b7c-43aa-8ad7-6fc071ad3f57" alt="Dataset mniset Confusion Matrix">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/7a445cff-a9a7-452f-88be-c89230c3392e" alt="Dataset mniset Loss Graph">
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/c3fb9284-6a92-4ec1-92eb-b70298af7b05" alt="Dataset mniset ROC Curve">
    </td>
  </tr>
</table>


For the CIFAR-10 dataset it can be said that model overfitted and could not achieved higher accuracy higher than %44 accuracy in 2400 epoch. This is due to complex data inside of CIFAR-10 and indicates that PCA, and regularization is not enough to reduce the complexity of the model and select features. Convolutional based approach is necessary and will be implemented in the final report.

## Visual Samples

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/fcecec27-fb92-4fe8-9e6c-c54e8c8705a5" alt="Dataset cifar10 Visual Samples" >
    </td>
    <td align="center">
      <img src="https://github.com/baturalpguven/Neural-Network-From-Scratch-Image-Classifier-for-CIFAR10-and-MNISET/assets/77858949/87c1222f-2c38-432c-85fc-601589ed03b6" alt="Dataset mniset Visual Samples" >
    </td>
  </tr>
</table>


## Running the Code
To run the code simply run the following in the command line for training.

```
python3 train.py
```

For evaluating the model run the following line for cifar10.

```
python3 train.py -- Train False --dataset cifar10
```
For evaluating the model run the following line for <a href="https://github.com/wouterkool/MNISET"> MNISET </a>.

```
python3 train.py -- Train True --dataset mniset
```


Replace `train.py` with the name of your script or executable. The `--dataset`, `--save_freq`,`--PCA`, `--PVE`, ` load_epoch ` and `--train` are example command-line arguments that you can customize based on your specific use case.

Command Line Arguments:

--dataset <str> (default: 'mniset')
    Name of the dataset to be used for training and evaluation.

--save_freq <int> (default: 200)
    Frequency of saving checkpoints during training.

--train <str> (default: 'true')
    Flag to indicate whether to start the training process. Set to 'true' to initiate training.

--PCA <str> (default: 'custom')
    Implementation of PCA for feature reduction. Choose from available options: 'custom', 'sklearn', 'pca_toolkit'.

--PVE <str> (default: 'false')
    Flag to calculate the Proportion of Variance Explained (PVE). Set to 'true' to enable PVE calculation.

--load_epoch <str> (default: '2400')
    Specify the epoch number to load a pre-trained model checkpoint for further training or evaluation.


Feel free to modify the example command and arguments to match the specific command line interface of your project.



## Referances

1. <a href="https://nthu-datalab.github.io/ml/index.html"> CS565600 Deep Learning, National Tsing Hua University </a>
2. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/"> Building a Neural Network from Scratch: Part 1 </a>
3. <a href="https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/"> Building a Neural Network from Scratch: Part 2 </a>
4. <a href="https://developer.ibm.com/technologies/artificial-intelligence/articles/neural-networks-from-scratch/"> Neural networks from scratch, IBM Developer</a>
5. <a href="https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/"> The Softmax Function Derivative (Part 1) </a>
6. <a href="https://github.com/lionelmessi6410/Neural-Networks-from-Scratch"> Neural-Networks-from-Scratch </a>








