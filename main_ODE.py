import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
import tensorflow_probability as tfp
import ast, argparse, time, os, sys, scipy.io, numpy as np

from scipy.interpolate import griddata
from pyDOE import lhs
from tqdm import tqdm
from sklearn.metrics import r2_score

sys.path.append('src/')
from ode_utils import PhysicsInformedNN
np.random.seed(1234); tf.random.set_seed(1243)
'''      Define the hyper-parameter
============================================'''
parser = argparse.ArgumentParser(description="ODE PINN")
parser.add_argument("--option", default='Prob1_2', type=str)
parser.add_argument("--gamma",  default=1, type=float)
parser.add_argument("--fig_dir", default='figures', type=str)
parser.add_argument("--NN_layers", default=[1, 20, 20, 1], type=ast.literal_eval)
parser.add_argument("--actv_func", default='tanh', type=str)
args = parser.parse_args()

class ODE_PINN:
    def __init__(self, option='Prob1_2',
                 fig_dir='figures', 
                 NN_layers=[1, 20, 20, 1],
                 actv_func='tanh'):
        self.data = None
        self.option = option
        self.fig_dir = fig_dir
        self.NN_layers = NN_layers
        self.actv_func = actv_func
        os.makedirs(self.fig_dir, exist_ok=True)

    def define_hyper_parameters(self):
        # # FOR Q2.1 and 2.2, please use these parameters (start)
        if self.option == 'Prob1_2':
            self.layers = [1, 20, 1]
            self.learning_rate = 0.5
            self.niter = 1000
            self.gamma = 1
            # # FOR Q2.1 and 2.2, please use these parameters (end)
        elif self.option == 'Prob3':
            # FOR Q2.3, please use these parameters (start)
            # self.layers = [1, 20, 20, 1]
            self.layers = self.NN_layers
            self.learning_rate = 0.04
            self.niter = 6000
            self.gamma = 1
            # FOR Q2.3, please use these parameters (end)

    def prepare_data_for_training(self):
        '''Preparing the data for training / validation
        ============================================'''
        ### Please do not change anything (Start)
        self.N_validation = 401
        self.x_validation = np.linspace(0, 2, self.N_validation)[:,None]
        self.u_validation = np.exp(self.x_validation)
        # Domain bounds
        self.lb = self.x_validation.min()
        self.ub = self.x_validation.max()
        self.data_type = tf.float32
        ### Please do not change anything (End)

        ### For Q2.1 and 2.2 (Start)
        self.N_collocation = 10 # We start with 10 collocation points. You might want to change number of collocation points to answer the questions
        self.x_train = tf.constant([0.], dtype=self.data_type)[:,None]
        self.u_train = tf.constant([1.], dtype=self.data_type)[:,None]
        self.x_f_train = self.lb + (self.ub-self.lb)*lhs(1, self.N_collocation)
        self.x_f_train = np.vstack((self.x_f_train, self.x_train))
        # Note, above line, we add the datapoint into the collocation point set, not neccessary,
        self.x_f_train = tf.cast(self.x_f_train, dtype=self.data_type) #transform the numpy array x_f_train into a tensor
        ### For Q2.1 and 2.2 (End)
        if self.option == 'Prob3':
            self.x_train = np.linspace(0, 2, 10)[:,None] 
            self.x_train = tf.cast(self.x_train, dtype=self.data_type)
            # ...10 data points evenly spaced in x ∈ [0,2]. (float32)
            self.u_train = np.exp(self.x_train) + 0.4*np.random.randn(10,1)
            self.u_train = tf.cast(self.u_train, dtype=self.data_type)
            # ...Gaussian noise with 0.4 standard deviation to uid (float32)

    def setup_model_and_plot_points(self):
        # Define the model
        self.model = PhysicsInformedNN(x_u=self.x_train, u=self.u_train, x_f=self.x_f_train,
                                x_val=tf.constant(self.x_validation, dtype=self.data_type)[:,None],
                                layers=self.layers, lb=self.lb, ub=self.ub, gamma=self.gamma, actv_func=self.actv_func)

        # Visualizing the data points / collocation points / valdiation points
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(self.x_validation, np.exp(self.x_validation), s=1, label='validation points')
        ax.scatter(self.x_f_train, np.exp(self.x_f_train), s=20, color='red', marker='*', label='collocation points')
        ax.scatter(self.x_train, self.u_train, s=20, color='black', marker='*', label='data points')
        ax.set_xlabel('x'); ax.set_ylabel('U')
        plt.legend(); plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/data_points_ode_{self.option}_gam{self.gamma}.png', dpi=300)

    def train_and_eval_model(self):
        start_time = time.time()
        self.it_list, self.loss_list, self.eq_loss_list, self.val_it_list, self.val_loss_list = self.model.train(self.niter, self.learning_rate)
        self.elapsed = time.time() - start_time
        tqdm.write('Training time: %.4f' % (self.elapsed))
        self.u_pred_val, self.f_pred_val = self.model.predict(tf.cast(self.x_validation, dtype=self.data_type))

    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(self.it_list, self.loss_list, label='training loss (total)')
        ax.plot(self.it_list, self.eq_loss_list, label='training loss (eq)')
        ax.plot(self.val_it_list, self.val_loss_list, label='validation loss (eq)')
        ax.set_title('Training Loss Curve'); ax.set_yscale('log')
        ax.set_xlabel('iteration'); ax.set_ylabel('loss')
        plt.legend(); plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/loss_ode_{self.option}_gam{self.gamma}.png', dpi=300)

    def plot_soln_and_residual(self):
        fig = plt.figure(figsize = [7.5, 7.5], dpi=300)

        ax = plt.subplot(211)
        ax.plot(self.x_validation, self.u_validation, 'b-', linewidth = 2, label = 'exact (validation points)')
        ax.plot(self.x_validation, self.u_pred_val, 'r--', linewidth = 2, label = 'predict (validation points)')
        ax.scatter(self.x_f_train, np.exp(self.x_f_train), marker='*', s=50, label = 'collocation points')
        ax.scatter(self.x_train, self.u_train, marker='*', s=50, label = 'data points')
        ax.set_xlabel(r'$x$', fontsize=15)
        ax.set_ylabel(r'$u$', fontsize=15, rotation=0)
        ax.set_title('solution', fontsize=15)

        ax = plt.subplot(212)
        ax.plot(self.x_validation, self.f_pred_val, 'b-', linewidth = 2)
        ax.scatter(self.x_f_train, np.zeros(len(self.x_f_train)), marker='*', s=50, label = 'collocation points', alpha=0.8)
        ax.scatter(self.x_train, np.zeros(len(self.x_train)), marker='*', s=50, label = 'data points', alpha=0.8)
        ax.set_xlabel(r'$x$', fontsize=15)
        ax.set_ylabel(r'$f$', fontsize=15, rotation=0)
        ax.set_title('equation residual', fontsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/residual_ode_{self.option}_gam{self.gamma}.png', dpi=300)
    
    def plot_scatter(self):
        self.R2 = r2_score(self.u_validation, self.u_pred_val)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(self.u_validation, self.u_pred_val, color='darkblue', label=rf'Data: $R^2$ = {self.R2:.3f}', alpha=0.75)
        ax.plot([self.u_validation.min(), self.u_validation.max()], 
                [self.u_validation.min(), self.u_validation.max()], 'r--', lw=2, label='Exact')
        ax.set_xlabel('Exact'); ax.set_ylabel('Prediction')
        plt.legend(); plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/scatter_plot_ode_{self.option}_gam{self.gamma}.png', dpi=300)


if __name__ == '__main__':
    pinn = ODE_PINN(option=args.option, fig_dir=args.fig_dir,
                    NN_layers=args.NN_layers, actv_func=args.actv_func)
    pinn.define_hyper_parameters()
    pinn.gamma = args.gamma
    pinn.prepare_data_for_training()
    pinn.setup_model_and_plot_points()
    pinn.train_and_eval_model()
    pinn.plot_loss()
    pinn.plot_soln_and_residual()
    pinn.plot_scatter()