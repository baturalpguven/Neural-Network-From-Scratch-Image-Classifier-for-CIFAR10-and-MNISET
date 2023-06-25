############## Adama implemantaion
## one more layer
import numpy as np
import time

class DNN():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
        
        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}
        
    def relu(self, x, derivative=False):

        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):

        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):

        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        hidden_layer_2=self.sizes[2]
        output_layer=self.sizes[3]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(hidden_layer_2, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((hidden_layer_2, 1)) * np.sqrt(1./hidden_layer),
            "W3": np.random.randn(output_layer, hidden_layer_2) * np.sqrt(1./hidden_layer_2),
            "b3": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer_2)
        }

        return params
    
    def initialize_momemtum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
            "W3": np.zeros(self.params["W3"].shape),
            "b3": np.zeros(self.params["b3"].shape),
        }
      
        return momemtum_opt

    def feed_forward(self, x):
        '''
            y = σ(wX + b)
        '''
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.activation(self.cache["Z2"])
        self.cache["Z3"] = np.matmul(self.params["W3"], self.cache["A2"]) + self.params["b3"]
        self.cache["A3"] = self.softmax(self.cache["Z3"])


        return self.cache["A3"]

    def Lasso(self,x):
        if x.all()>0:
            x=1
        elif x.all()<0:
            x=-1
        else:
            x=0
        return x
    
    def back_propagate(self, y, output):
 
        current_batch_size = y.shape[0]
        lambda_reg = 0.05

        dZ3 = output - y.T
        dW3 = (1./current_batch_size) * np.matmul(dZ3, self.cache["A2"].T) + (lambda_reg/current_batch_size)*(self.params["W3"])
        db3 = (1./current_batch_size) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.matmul(self.params["W3"].T, dZ3)
        dZ2 = dA2 * self.activation(self.cache["Z2"], derivative=True)
        dW2 = (1./current_batch_size) * np.matmul(dZ2, self.cache["A1"].T) + (lambda_reg/current_batch_size)*(self.params["W2"])
        db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1./current_batch_size) * np.matmul(dZ1, self.cache["X"]) + (lambda_reg/current_batch_size)*(self.params["W1"])
        db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        # print("db2:",np.shape(db2))
        return self.grads

    def cross_entropy_loss(self, y, output):

        current_batch_size = y.shape[0]
        l_sum = np.sum(np.multiply(y.T, np.log(output))) - ( np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3'])))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l
                
    def optimize(self, l_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta1 * self.momemtum_opt[key] + (1. - beta1) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        elif self.optimizer == "adam":
            t = self.iterations
            for key in self.params:
                self.momemtum_opt[key] = (beta1 * self.momemtum_opt[key] + (1. - beta1) * self.grads[key])
                self.velocity_opt[key] = (beta2 * self.velocity_opt[key] + (1. - beta2) * self.grads[key]**2)
                m_hat = self.momemtum_opt[key] / (1 - beta1**(t+1))
                v_hat = self.velocity_opt[key] / (1 - beta2**(t+1))
                self.params[key] = self.params[key] - l_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd', 'momentum', or 'adam' instead.")
        self.iterations += 1

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
    

    def predict(self,x,params,confidance_rate):
        self.params = params
        output = self.feed_forward(x)
        prob = self.feed_forward(x)

        for i in range(len(output)):
            for pred in range(len(output[0])):
                if output[i][pred] > confidance_rate:
                    output[i][pred] = 1
                else:
                    output[i][pred] = 0
        pred = output
        return pred,prob

    def confusion_matrix(self,num_labels,y, yhat):
        conf_matrix = np.zeros((num_labels, num_labels))
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                conf_matrix[i][j] = np.sum((y == i) & (yhat == j))
        return conf_matrix

    def roc_curve(self,y, yhat, num_classes):
        thresholds = np.linspace(0, 1, 100)
        true_positive_rate = dict()
        false_positive_rate = dict()
        for one_class in range(num_classes):
            true_positive = np.zeros(len(thresholds))
            false_positive = np.zeros(len(thresholds))
            for i, threshold in enumerate(thresholds):
                yhat_th = yhat[:, one_class] > threshold
                true_positive[i] = np.sum((y[:, one_class] == 1) & (yhat_th == 1))
                false_positive[i] = np.sum((y[:, one_class] == 0) & (yhat_th == 1))
            true_positive_rate[one_class] = true_positive / np.sum(y[:, one_class])
            false_positive_rate[one_class] =  false_positive/ np.sum(1 - y[:, one_class])
        
        # Compute micro-average TPR and FPR
        tpr = np.sum(list(true_positive_rate.values()), axis=0) / num_classes
        fpr = np.sum(list(false_positive_rate.values()), axis=0) / num_classes
        
        # Compute micro-average AUC using trapezoidal rule
        auc = abs(np.trapz(tpr, fpr))
        
        return tpr, fpr, auc


    def train(self, x_train, y_train, x_test, y_test,x_val,y_val,save_freq, args, epochs, 
              batch_size, optimizer, l_rate, beta):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        train_acc_list,train_loss_list,test_acc_list,val_acc_list,val_loss_list = [],[],[],[],[]
        
        # Initialize optimizer
        self.optimizer = optimizer
        if self.optimizer == 'momentum' or self.optimizer == 'adam' :
            self.momemtum_opt = self.initialize_momemtum_optimizer()
            self.velocity_opt = self.initialize_momemtum_optimizer()
            self.iterations = 0
        
        start_time = time.time()
        # template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}"
        
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]


            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                # Forward
                output = self.feed_forward(x)
                # Backprop
                _ = self.back_propagate(y, output)
                # Optimize
                self.optimize(l_rate=l_rate, beta1=beta, beta2=0.999, epsilon=1e-8)

            # Evaluate performance
            # Training data
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_acc_list.append(train_acc)
            train_loss = self.cross_entropy_loss(y_train, output)
            train_loss_list.append(train_loss)

            # Validate data

            output3 = self.feed_forward(x_val)
            val_acc = self.accuracy(y_val, output3)
            val_acc_list.append(val_acc)
            val_loss = self.cross_entropy_loss(y_val, output3)
            val_loss_list.append(val_loss)

            # Test data
            output2 = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output2)
            test_acc_list.append(test_acc)

            if i % save_freq == 0 and i !=0 :
                np.save("./checkpoints/Epoch_{}_{}".format(i,args.dataset),self.params)
                np.save("./checkpoints/Epoch_{}_{}_train_acc".format(i,args.dataset), np.array(train_acc_list, dtype=object))
                np.save("./checkpoints/Epoch_{}_{}_train_loss".format(i,args.dataset),np.array(train_loss_list, dtype=object))
                np.save("./checkpoints/Epoch_{}_{}_val_acc".format(i,args.dataset), np.array(val_acc_list, dtype=object))
                np.save("./checkpoints/Epoch_{}_{}_val_loss".format(i,args.dataset),np.array(val_loss_list, dtype=object))
                np.save("./checkpoints/Epoch_{}_{}_test_acc".format(i,args.dataset),np.array(test_acc_list, dtype=object))
                print("-------------------Epoch_{}_{} Saved-------------------".format(i,args.dataset))

            print("Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f},val acc={:.2f}, val loss={:.2f},test acc={:.2f}"
                  .format(i+1, time.time()-start_time, train_acc, train_loss,val_acc,val_loss, test_acc))
            # print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc))
