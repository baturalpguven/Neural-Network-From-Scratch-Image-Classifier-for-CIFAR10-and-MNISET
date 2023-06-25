import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from model import DNN
import pandas as pd
from mniset import load_mniset, extract_grayscale_dataset
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

## These argumants are necessary to run this code through terminal. 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mniset', help='Name of the dataset')
parser.add_argument('--save_freq' ,type=int, default=200, help='saving frequency')
parser.add_argument('--train' ,type=str, default="true", help='start training')
parser.add_argument('--PCA', type=str, default='custom', help='My PCA Implemantation')
parser.add_argument('--PVE', type=str, default='false', help='Calculate PVE')
parser.add_argument('--load_epoch', type=str, default='2400', help='Choose which epoch to load')
args = parser.parse_args()
## All of these parameters have default values but can be changed whenever needed.
    
### OneHotEncoding_encoding of the labels for     
def OneHotEncoding(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)
### necessary number of classes.
def main():

    if args.dataset == "cifar10":
        ################################ Load CIFAR-10 dataset
        print("Loading training data...")
        x = []
        y = []
        for num in range(5):
            cifar = pd.read_pickle(r'./CIFAR_10/train/cifar-10-batches-py/data_batch_{}'.format(num+1))
            x_batch = cifar["data"]
            y_batch = cifar["labels"]
            y_batch = np.array(y_batch)
            x.append(x_batch)
            y.append(y_batch)
        x = np.concatenate(x)
        y = np.concatenate(y)

        print("x_train shape:",x.shape)
        print("y_train shape:",y.shape)

        print("Loading test data...")
        x_test = []
        y_test = []
        cifar = pd.read_pickle(r'./CIFAR_10/test/cifar-10-batches-py/test_batch')
        x_batch_test = cifar["data"]
        y_batch_test = cifar["labels"]
        y_batch_test = np.array(y_batch_test)
        x_test.append(x_batch_test)
        y_test.append(y_batch_test)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)
        

        test_img_x = x_test
        test_label_y = y_test

        print("x_test shape:",x_test.shape)
        print("y_test shape:",y_test.shape)

        num_labels =  10
        features = 250
        print("-----------------------CIFAR-10 dataset used-----------------------")
    
    ############################### Load MNISET dataset
    if args.dataset == "mniset":
        mniset = load_mniset()
        x, y, *_ = extract_grayscale_dataset(mniset, split='train',fill=True)
        x_test, y_test, *_ = extract_grayscale_dataset(mniset, split='test',fill=True)

        test_img_x, test_label_y, labels, _ = extract_grayscale_dataset(mniset, split='test',fill=True)
        
        num_labels =  27
        features = 150
        print("-----------------------MNISET dataset used-----------------------")

    # Normalize
    print("Preprocessing data...")
    if args.dataset == "cifar10":
        ######### convert gray scale
        x = x.reshape(x.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x = 0.21*x[:,:,:,0]/255 + 0.72*x[:,:,:,1]/255 + 0.07*x[:,:,:,2]/255
        ######### convert gray scale
    x = x - np.mean(x)
    x = np.clip(x,-3*np.std(x),3*np.std(x))
    x = x + 0.5
    x_dims = x.shape[1]**2
    x = x.reshape((x.shape[0],x.shape[1]**2))

    #################### PCA
    if args.PCA == "custom":
        covMatrix = np.cov(x.T)
        eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
        if args.PVE == "true":
            PVE = []
            for i in range(x_dims):
                pve = np.sum(eigenvalues[:i+1])*100 / np.sum(eigenvalues)
                PVE.append(pve)
            plt.figure()
            print()
            plt.plot(range(x_dims),PVE)
            plt.xlabel("# Features")
            plt.ylabel("Variance %")
            plt.title('Dataset {} PVE'.format(args.dataset))
            plt.savefig("./results/Dataset {} PVE Graph.png".format(args.dataset))
            print("-----------------------PVE Generated-----------------------")

        x =  np.dot(x,eigenvectors[:, :features])
    else:
        pca_100 = PCA(n_components=features)
        pca_100.fit(x)
        x = pca_100.transform(x)

    ##############################
    if args.dataset == "cifar10":
        ######### convert gray scale
        x_test = x_test.reshape(x_test.shape[0],3, 32, 32).transpose(0, 2, 3, 1)
        x_test = 0.21*x_test[:,:,:,0]/255 + 0.72*x_test[:,:,:,1]/255 + 0.07*x_test[:,:,:,2]/255


        test_img_x = test_img_x.reshape(test_img_x.shape[0],3, 32, 32).transpose(0, 2, 3, 1)
        test_img_x = 0.21*test_img_x[:,:,:,0]/255 + 0.72*test_img_x[:,:,:,1]/255 + 0.07*test_img_x[:,:,:,2]/255

        ######### convert gray scale
    x_test = x_test - np.mean(x_test)
    x_test = np.clip(x_test,-3*np.std(x_test),3*np.std(x_test))
    x_test = x_test + 0.5

    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]**2))

        ##################### PCA
    if args.PCA == "custom":
        x_test =  np.dot(x_test,eigenvectors[:, :features])
    else:
        x_test = pca_100.transform(x_test)
    ##############################

    # One-hot encode labels
    y_new = OneHotEncoding(y.astype('int32'), num_labels)
    
    y_test_last = OneHotEncoding(y_test.astype('int32'), num_labels)

    # Split, reshape, shuffle
    train_size =  round(y.shape[0]*0.8)
    val_size = y.shape[0] - train_size
    test_size = x_test.shape[0]
    x_train, x_val, x_test = x[:train_size], x[train_size:train_size+val_size], x_test[:test_size]
    y_train, y_val, y_test = y_new[:train_size], y_new[train_size:train_size+val_size], y_test_last[:test_size]
    shuffle_index = np.random.permutation(train_size)
    shuffle_index_val = np.random.permutation(val_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    x_val, y_val = x_val[shuffle_index_val], y_val[shuffle_index_val]
    print("Training data: {} {}".format(x_train.shape, y_train.shape))
    print("Validation data: {} {}".format(x_val.shape, y_val.shape))
    print("Test data: {} {}".format(x_test.shape, y_test.shape))

    # Train
    if args.train == "true":
        print("---------------Start training!---------------")  
        dnn = DNN(sizes=[features,64,32,num_labels], activation='sigmoid')
        dnn.train(x_train, y_train, x_test, y_test,x_val,y_val, args.save_freq,args, epochs=10000, batch_size=64, optimizer='adam', l_rate=0.0005, beta=.9) # good lr 0.2
    ### sgd|momentum|adam
    else:
        print("---------------Start Generating Evaluation Metrics---------------")
        dnn = DNN(sizes=[features, 64, 32, num_labels], activation='sigmoid')
        params = np.load("./checkpoints/Epoch_{}_{}.npy".format(args.load_epoch,args.dataset), allow_pickle=True)
        params_dict = dict(params.item())
        prediction,prob = dnn.predict(x_test,params_dict,0.4)
        yhat = prediction.T
        ################# conf_matrix
        # conf_matrix = confusion_matrix(y_test.argmax(axis=1), yhat.argmax(axis=1))
        conf_matrix = dnn.confusion_matrix(y_test.shape[1],y_test.argmax(axis=1), yhat.argmax(axis=1))
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues')
        plt.xlabel("Predicted Values")
        plt.ylabel("True Values")
        plt.title("Dataset {} Confusion Matrix".format(args.dataset))
        plt.savefig("./results/Dataset {} Confusion Matrix.png".format(args.dataset))
        print("Conf_matrix Generated !!!")

        ################# Error and Acc Curves
        train_acc = np.load("./checkpoints/Epoch_{}_{}_train_acc.npy".format(args.load_epoch,args.dataset), allow_pickle=True)
        train_loss = np.load("./checkpoints/Epoch_{}_{}_train_loss.npy".format(args.load_epoch,args.dataset), allow_pickle=True)
        val_acc = np.load("./checkpoints/Epoch_{}_{}_val_acc.npy".format(args.load_epoch,args.dataset), allow_pickle=True)
        val_loss = np.load("./checkpoints/Epoch_{}_{}_val_loss.npy".format(args.load_epoch,args.dataset), allow_pickle=True)
        test_acc = np.load("./checkpoints/Epoch_{}_{}_test_acc.npy".format(args.load_epoch,args.dataset), allow_pickle=True)

        plt.figure()
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.plot(test_acc)
        plt.legend(['Train Acc','Val Acc','Test Acc'])
        plt.title('Dataset {} Accuracy'.format(args.dataset))
        plt.xlabel('# Epochs')
        plt.ylabel('Acc')
        plt.savefig("./results/Dataset {} Acc Graph.png".format(args.dataset))
        
        plt.figure()
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(['Train Loss','Val Loss'])
        plt.title('Dataset {} Cross Entropy Loss'.format(args.dataset))
        plt.xlabel('# Epochs')
        plt.ylabel('Loss')
        plt.savefig("./results/Dataset {} Loss Graph.png".format(args.dataset))
        print("Acc and Loss Graps Generated !!!")

        ######################## Generate Samples
        plt.figure()
        if args.dataset == "cifar10":
            names = pd.read_pickle(r'./CIFAR_10/train/cifar-10-batches-py/batches.meta')['label_names']
            print(names)
        else:
            names = labels
        plt.figure()
        for index,img in enumerate(test_img_x[:9,:,:]):
            plt.subplot(3,3,index+1)
            plt.imshow(img,cmap='gray')
            plt.axis('off')
            plt.title("True Label: {} \n Predicted Label: {}".format(names[test_label_y[index]],names[yhat.argmax(axis=1)[index]]),fontsize=5)
        plt.subplots_adjust(wspace=0.3, hspace=0.5, bottom=0.2)
        plt.suptitle('Dataset {} Visual Samples'.format(args.dataset))
        plt.savefig("./results/Dataset {} Visual Samples.png".format(args.dataset))
        print("Visual Samples Generated !!!")

        ########################

        ######################## ROC Curve
        yhat = prob.T
        tpr,fpr,roc_auc = dnn.roc_curve(y_test,yhat,num_labels)
        ## Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Dataset {} ROC'.format(args.dataset))
        plt.legend(loc="lower right")

        # Save ROC curve
        plt.savefig("./results/Dataset {} ROC Curve.png".format(args.dataset))
        print("ROC Generated !!!")

if __name__ == '__main__':
    main()
"""
Note:
    2 hidden layer (32 64) with 0.05 reg and builtin PCA network generates 66 acc in test acc 65-66 4K epoch
    while (64 64 32 ) with 0.08 reg and PCA  acc 65-67 in 6.5K epochs however training and test acc are closer
    with hand crafted PCA it oscilates around 57-67

    ADAM 5K 67-68 acc for 2 hidde layer (64 32) lr 0.0005
    ADAM 2K 66-67 acc for 2 hidde layer (32 64) lr 0.0005
    ADAM 2K 66-67 acc for 2 hidde layer (64 64) lr 0.0005
"""