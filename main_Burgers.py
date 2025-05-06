import pyDOE, argparse, scipy.io, time, os, tensorflow as tf, numpy as np, sys
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import griddata
from pyDOE import lhs
from tqdm import tqdm
from sklearn.metrics import r2_score

sys.path.append('src/')
from burgers_utils import PhysicsInformedNN
np.random.seed(123); tf.random.set_seed(123)
print(tf.__version__)

parser = argparse.ArgumentParser(description="Burgers PINN")
parser.add_argument("--add_noise", action='store_true', default=False, help="Add noise to the data")
parser.add_argument("--actv_func", default='tanh', type=str, help="Activation function")
parser.add_argument("--fig_dir", default='figures', type=str, help="Directory to save figures")
parser.add_argument("--inverse_problem", action='store_true', default=False, help="Inverse problem")
parser.add_argument("--iterations", default=5000, type=int, help="Number of iterations for training")
parser.add_argument("--decay_steps", default=200, type=int, help="Decay steps for learning rate")
args = parser.parse_args()

class BurgersPINN:
    def __init__(self, iterations=5000, fig_dir='figures', 
                 add_noise=False, actv_func='tanh', 
                 decay_steps=200, inverse_problem=False):
        self.data = None
        self.iterations = iterations
        self.fig_dir = fig_dir
        self.add_noise = add_noise
        self.actv_func = actv_func
        self.decay_steps = decay_steps
        self.inverse_problem = inverse_problem
        os.makedirs(fig_dir, exist_ok=True)

    def get_data(self):
        self.data = scipy.io.loadmat('data/burgers_shock.mat')
        # Extract variables
        self.t = self.data['t'].squeeze()     # shape (100,)
        self.x = self.data['x'].squeeze()     # shape (256,)
        self.Exact = np.real(self.data['usol']).T        # shape (256, 100)
        print(f"x shape: {self.x.shape}, t shape: {self.t.shape}, usol shape: {self.Exact.shape}")

    @staticmethod
    def plot_2d_soln(x, t, U, X_u_train, actv_func='tanh',
                     name='exact', fig_dir='.', inverse_problem=False): # Plot the 2D field u(x, t)
        plt.figure(figsize=(8, 5))
        plt.imshow(U.T, interpolation='nearest', cmap='rainbow',
                extent=[t.min(), t.max(), x.min(), x.max()],
                origin='lower', aspect='auto')
        if inverse_problem:
            plt.scatter(X_u_train[:, 0], X_u_train[:, 1], marker='x',
                        color='black', s=10, label='Sampled Data Points')
        plt.xlabel('Time t')
        plt.ylabel('x')
        plt.title("Solution to 1D Burgers' Equation")
        plt.colorbar(label=r'$u(x,t)$')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{fig_dir}/1D_Burgers_{name}_actv_{actv_func}.png', dpi=300)

    @staticmethod
    def plot_1d_soln(t_tar, x, t, U1, U2, actv_func='tanh',
                     key1='Exact', key2='Prediction', fig_dir='.'): # Plot the 2D field u(x, t)
        idx = np.where(t == t_tar)[0]
        plt.figure(figsize=(5, 5))
        plt.plot(x, U1[idx,:].flatten(), color='darkblue', label=key1)
        plt.plot(x, U2[idx,:].flatten(), '--', color='red', label=key2)
        plt.xlabel('Time t')
        plt.ylabel('x')
        plt.title(f"Solution to 1D Burgers' Equation @ t={t_tar}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{fig_dir}/1DSlice_Burgers_t{t_tar}_actv_{actv_func}.png', dpi=300)

    def setup_data(self):
        # train the model
        self.nu = 0.01/np.pi
        if self.add_noise:
            noise = 0.05*np.random.randn(self.Exact.shape[0], self.Exact.shape[1])
            self.Exact += noise
        self.N_u, self.N_f = 100, 10000 # n
    
        self.X, self.T = np.meshgrid(self.x, self.t) # X, T dimensions: 100 x 256
        self.X_star = np.hstack((self.X.flatten()[:,None], self.T.flatten()[:,None])) # dimension of 25600 x 2
        self.u_star = self.Exact.flatten()[:,None] # dimension of 25600 x 1

        # Domain bounds
        self.lb = self.X_star.min(0)
        self.ub = self.X_star.max(0)

        xx1 = np.hstack((self.X[0:1,:].T, self.T[0:1,:].T))   # x,t coordinates of the training data at t=0
        uu1 = self.Exact[0:1,:].T
        xx2 = np.hstack((self.X[:,0:1], self.T[:,0:1]))       # x,t coordinates of the training data at x=1
        uu2 = self.Exact[:,0:1]
        xx3 = np.hstack((self.X[:,-1:], self.T[:,-1:]))       # x,t coordinates of the training data at x=-1
        uu3 = self.Exact[:,-1:]
        self.u_train = np.vstack([uu1, uu2, uu3])

        if self.inverse_problem:
            # self.X_u_train = lhs(2, samples=self.N_u) # https://pythonhosted.org/pyDOE/randomized.html
            self.bounds = np.array([[self.t.min(), self.t.max()],[self.x.min(), self.x.max()]]) #[Bounds]
            self.X_u_train = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.N_u, 2))
            # this is the key code for problem 1.5
            # bounds = np.array([[-10,10],[1,20]]) #[Bounds] https://stackoverflow.com/questions/66977657/latin-hypercube-sampling
        else:
            self.X_u_train = np.vstack([xx1, xx2, xx3])      # x,t coordinates of all training data at t=0, x=1, x=-1
        self.X_f_train = self.lb + (self.ub - self.lb)*lhs(2, self.N_f) # x,t coordinates of all collocation points
        # LHS: Latin Hypercube Sampling, https://pythonhosted.org/pyDOE/randomized.html
        # This youtube video may help you with understanding what Latin Hypercube Sampling is
        # https://www.youtube.com/watch?v=ugy7XC-cMb0

        idx = np.random.choice(self.X_u_train.shape[0], self.N_u, replace=False)
        self.X_u_train = self.X_u_train[idx, :]
        self.u_train   = self.u_train[idx,:]

        self.X_u_train = tf.cast(self.X_u_train, dtype=tf.float32)
        self.u_train   = tf.cast(self.u_train,   dtype=tf.float32)
        self.X_f_train = tf.cast(self.X_f_train, dtype=tf.float32)

    def train_model(self):
        self.layers = [2, 20, 20, 20, 1]
        self.model = PhysicsInformedNN(self.X_u_train, self.u_train, self.X_f_train,
                                       self.layers, self.lb, self.ub,
                                       self.nu, actv_func=self.actv_func, 
                                       decay_steps=self.decay_steps)

        start_time = time.time()
        self.it_list, self.loss_list = self.model.train(self.iterations, learning_rate=5e-2)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))

    def eval_model(self): # Make prediction on the mesh of the entire field
        self.X_star = tf.cast(self.X_star, dtype=tf.float32)
        self.u_pred, self.f_pred = self.model.predict(self.X_star)

        self.error_u = np.linalg.norm(self.u_star - self.u_pred,2)/np.linalg.norm(self.u_star,2)
        print('Error u: %e' % (self.error_u))

        self.U_pred = griddata(self.X_star, self.u_pred.numpy().flatten(), 
                          (self.X, self.T), method='cubic')
        self.Error = np.abs(self.Exact - self.U_pred)
        self.R2 = r2_score(self.Exact, self.U_pred)

    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(self.it_list, self.loss_list, color='darkblue')
        ax.set_title('Training Loss Curve')
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        plt.tight_layout()
        if self.inverse_problem:
            plt.savefig(f'{self.fig_dir}/loss_burgers_actv{self.actv_func}_inverse_prob.png', dpi=300)
        else:
            plt.savefig(f'{self.fig_dir}/loss_burgers_actv{self.actv_func}_it{self.iterations}.png', dpi=300)

    def scatter_plot(self):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(self.Exact, self.U_pred, color='darkblue', 
                   label=rf'Data: $R^2$ = {self.R2:.3f}', alpha=0.5)
        ax.plot([self.Exact.min(), self.Exact.max()], 
                [self.Exact.min(), self.Exact.max()], 'r--', lw=2, label='Exact')
        ax.set_xlabel('Exact'); ax.set_ylabel('Prediction')
        plt.legend(); plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/scatter_plot_burgers_actv{self.actv_func}.png', dpi=300)


if __name__ == '__main__':
    pinn = BurgersPINN(iterations=args.iterations, add_noise=args.add_noise, 
                       actv_func=args.actv_func, fig_dir=args.fig_dir,
                       decay_steps=args.decay_steps, inverse_problem=args.inverse_problem)
    pinn.get_data()
    pinn.setup_data()
    pinn.train_model()
    pinn.eval_model()
    pinn.plot_loss()
    pinn.plot_2d_soln(pinn.x, pinn.t, pinn.Exact, pinn.X_u_train, actv_func=pinn.actv_func,
                      name='exact', fig_dir=pinn.fig_dir, inverse_problem=pinn.inverse_problem)
    pinn.plot_2d_soln(pinn.x, pinn.t, pinn.U_pred, pinn.X_u_train, actv_func=pinn.actv_func,
                      name='pred',  fig_dir=pinn.fig_dir, inverse_problem=pinn.inverse_problem)
    pinn.plot_2d_soln(pinn.x, pinn.t, pinn.Error, pinn.X_u_train, actv_func=pinn.actv_func,
                      name='error', fig_dir=pinn.fig_dir, inverse_problem=pinn.inverse_problem)
    pinn.plot_1d_soln(0.5, pinn.x, pinn.t, pinn.Exact, pinn.U_pred, fig_dir=pinn.fig_dir)
    pinn.scatter_plot()