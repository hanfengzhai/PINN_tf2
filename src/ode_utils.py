import time, tensorflow as tf, numpy as np
from tqdm import tqdm

class PhysicsInformedNN:
    # Initialize the class with training and validation data, network structure, and parameters
    def __init__(self, x_u, u, x_f, x_val, 
                 layers, lb, ub, gamma, actv_func='tanh'):
        self.actv_func = actv_func
        # Lower and upper bounds of the input domain (used for normalization)
        self.lb = lb
        self.ub = ub

        # Supervised training data: locations and known values
        self.x_u = x_u  # Collocation points with known solution
        self.u = u      # Known solution values at x_u

        # Collocation points for enforcing the physics (residual of ODE)
        self.x_f = x_f

        # Validation points (used for monitoring generalization)
        self.x_val = x_val

        # Neural network architecture: list of neurons per layer
        self.layers = layers

        # Weight for the ODE loss (residual) in total loss
        self.gamma = gamma

        # Initialize network weights and biases
        self.weights, self.biases = self.initialize_NN(layers)

        # List of all trainable variables (used by optimizer)
        self.train_variables = self.weights + self.biases

        # # Compute initial loss (not strictly necessary here, could be deferred)
        # self.loss = self.loss_NN()

    '''
    Neural Network Initialization
    =============================
    '''

    # Initialize the weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Xavier initialization helps maintain signal variance across layers
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        """
        Function to construct the forward pass of the neural network
        Input:
        1. X: input tensor
        2. weights: All W matrixes, NN parameters
        3. biases: All b matrixes, NN parameters
        Output:
        1. Y: NN prediction
        """
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 #normalize the input to [-1, 1]
        #########################
        #########################
        ### YOUR CODE STARTS HERE (~7 lines, 6 points)
        ### Note: we do not apply activation function for the last layer
        # Hidden layers with tanh activation
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            if self.actv_func == 'tanh':
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            elif self.actv_func == 'relu':
                H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            elif self.actv_func == 'sigmoid':
                H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
            elif self.actv_func == 'exp':
                H = tf.exp(tf.add(tf.matmul(H, W), b))
                

        ### YOUR CODE ENDS HERE
        #########################
        #########################
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    '''
    Physics-Informed Loss Construction
    ==================================
    '''

    # Predict the solution u(x) using the neural network
    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    #
    def net_f(self, x):
        """
        Function to Compute the ODE residual f = du/dx - u using automatic differentiation
        Output:
        1. f: equation residual
        """
        ### More details of tf.GradientTape() you can follow
        ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.net_u(x)
            ### YOUR CODE STARTS HERE (~2 lines, 3 points)
            u_x = tape.gradient(u, x)
        f = u_x - u
        ### YOUR CODE ENDS HERE
        return f

    @tf.function
    def loss_NN(self):
        """
        Function to compute the total loss: data loss + equation loss (weighted by gamma)
        Output:
        1. loss_d: data loss
        2. loss_e: equation loss
        3. loss: total loss
        """
        #########################
        #########################
        ### YOUR CODE STARTS HERE (~4 lines, 6 points)
        ### Hint: Use mean squared loss for both equation loss and data loss
        self.u_pred = self.net_u(self.x_u)
        self.f_pred = self.net_f(self.x_f)

        loss_d = tf.reduce_mean(tf.square(self.u - self.u_pred))
        loss_e = tf.reduce_mean(tf.square(self.f_pred))

        ### YOUR CODE ENDS HERE
        #########################
        #########################
        ### Congratulations, if you fill in the code blocks above right, you will able to
        ### reproduce the plots below simply running through the rest of the code blocks
        loss = loss_d + self.gamma * loss_e
        return loss, loss_d, loss_e

    # Train the model using Adam optimizer
    def train(self, nIter: int, learning_rate: float):
        """
        Function used for training the model using the Adam optimizer.
        Implements exponential decay for learning rate.
        Tracks training and validation loss at intervals.
        """

        # Learning rate schedule: exponential decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=400,
            decay_rate=0.8)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        start_time = time.time()
        it_list = []          # Track training iterations
        loss_list = []        # Track total loss
        eq_loss_list = []     # Track equation loss
        val_it_list = []      # Validation iteration steps
        val_loss_list = []    # Validation loss values

        for it in tqdm(range(nIter)):
            with tf.GradientTape() as tape:
                loss, loss_d, loss_e = self.loss_NN()
                grads = tape.gradient(loss, self.train_variables)
                optimizer.apply_gradients(zip(grads, self.train_variables))

                it_list.append(it)
                loss_list.append(loss)
                eq_loss_list.append(loss_e)

            # Every 10 steps, evaluate validation loss and print status
            if (it + 1) % 10 == 0:
                residual = self.net_f(self.x_val)
                val_loss = tf.reduce_mean(tf.square(residual)) * self.gamma
                val_it_list.append(it)
                val_loss_list.append(val_loss)

                elapsed = time.time() - start_time
                tqdm.write('It: %d, Train Loss: %.4e, Data Loss: %.4e, Eq Loss: %.4e, Val Loss: %.4e Time: %.2f' %
                        (it, loss.numpy(), loss_d.numpy(), loss_e.numpy(), val_loss.numpy(), elapsed))
                start_time = time.time()

        return it_list, loss_list, eq_loss_list, val_it_list, val_loss_list

    # Predict u and residual f for new input points
    @tf.function
    def predict(self, x):
        u_p = self.net_u(x)
        f_p = self.net_f(x)
        return u_p, f_p