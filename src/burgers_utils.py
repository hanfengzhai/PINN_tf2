import time, tensorflow as tf, numpy as np
from tqdm import tqdm

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, 
                 layers, lb, ub, nu, 
                 actv_func='tanh', decay_steps=200):
        self.actv_func = actv_func
        self.decay_steps = decay_steps
        # Domain bounds for input normalization
        self.lb = lb
        self.ub = ub

        # Split supervised data inputs (with known solution u) into x and t
        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        # Split collocation points (where PDE is enforced) into x and t
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        # Known solution at supervised points
        self.u = u

        # Network architecture and PDE parameter
        self.layers = layers
        self.nu = nu

        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_NN(layers)

        # All trainable parameters for optimization
        self.train_variables = self.weights + self.biases
        # These are tf.Variable objects so updates here affect the model directly

        # Initialize loss (this is not necessary here but allows pre-evaluation)
        self.loss = self.loss_NN()

    '''
    Neural Network Initialization Functions
    =======================================
    '''

    def initialize_NN(self, layers):
        # Initialize weights and biases for each layer using Xavier init
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        # Xavier/Glorot initialization to avoid vanishing/exploding gradients
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        # Forward pass through the fully connected feedforward neural network
        num_layers = len(weights) + 1

        # Normalize inputs to [-1, 1] using domain bounds
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

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
            elif self.actv_func == 'sin':
                H = tf.sin(tf.add(tf.matmul(H, W), b))

        # Output layer (linear activation)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    '''
    Physics-Informed PDE Modeling
    =============================
    '''

    def net_u(self, x, t):
        # Predict u(x, t) by feeding concatenated input into the network
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        # Construct the PDE residual f(x, t) = u_t + u*u_x - nu*u_xx using autograd
        X = tf.concat([x, t], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])

            # Forward pass: predict u(x,t)
            u = self.net_u(x, t)

            # Compute gradients: first-order
            u_x = tape.gradient(u, x)
            u_t = tape.gradient(u, t)

            # Compute second-order gradient: u_xx
            u_xx = tape.gradient(u_x, x)

        # Burgers' equation residual
        f = u_t + u * u_x - self.nu * u_xx
        return f

    @tf.function
    def loss_NN(self):
        # Compute the total loss: data loss + PDE residual loss

        # Prediction of u at supervised data points
        self.u_pred = self.net_u(self.x_u, self.t_u)

        # Residual of PDE at collocation points
        self.f_pred = self.net_f(self.x_f, self.t_f)

        # Mean squared error between predicted and known u
        loss_data = tf.reduce_mean(tf.square(self.u - self.u_pred))

        # Mean squared error of PDE residual
        loss_eq = tf.reduce_mean(tf.square(self.f_pred))

        # Total loss = data loss + physics loss (unweighted here)
        return loss_data + loss_eq

    def train(self, nIter: int, learning_rate: float):
        """
        Train the model using Adam optimizer and gradient clipping.
        Uses exponential decay learning rate and logs progress every 100 iterations.
        """
        # Exponential learning rate decay, to help release the issue of training loss oscilation
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        varlist = self.weights + self.biases  # All trainable variables

        # Logging variables
        start_time = time.time()
        it_list = []
        loss_list = []

        for it in tqdm(range(nIter)):
            with tf.GradientTape() as tape:
                loss = self.loss_NN()

            # Compute gradients of loss w.r.t. weights and biases
            grads = tape.gradient(loss, varlist)

            # Optional: clip gradients to prevent exploding updates
            clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

            # Apply gradients to update model parameters
            optimizer.apply_gradients(zip(clipped_grads, varlist))

            # Logging
            it_list.append(it)
            loss_list.append(loss)

            if (it + 1) % 100 == 0:
                elapsed = time.time() - start_time
                tqdm.write('It: %d, Train Loss: %.3e, Time: %.2f' % (it, loss.numpy(), elapsed))
                start_time = time.time()

        return it_list, loss_list

    @tf.function
    def predict(self, X_star):
        # Evaluate the trained model: predict both u and f at new points
        u_star = self.net_u(X_star[:, 0:1], X_star[:, 1:2])
        f_star = self.net_f(X_star[:, 0:1], X_star[:, 1:2])
        return u_star, f_star