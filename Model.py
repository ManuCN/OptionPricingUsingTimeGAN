#! pip install ydata-synthetic --user

from os import path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
import random
import seaborn as sns

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

plt.rcParams["font.family"] = "Times New Roman"

class Model():
    def __init__(self, seq_len, hidden_dim, noise_dim, dim, batch_size, learning_rate, data_ret_dir, data_abs_dir, n_seq=5, column_of_interest=3):
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.dim = dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.columns = ["PX_OPEN","PX_HIGH","PX_LOW","PX_LAST","PX_VOLUME"]
        self.column_of_interest = column_of_interest
        self.__e_func = np.vectorize(self.__e)
        self.data_ret_dir = data_ret_dir
        self.data_abs_dir = data_abs_dir
        self.gamma = 1
        
    def load_model(self, model_dir):
        self.model_dir = model_dir
        self.synth = TimeGAN.load(model_dir)
        self.__set_real_min_max(self.data_ret_dir)
        
    def train_model(self, epochs, model_name, save=True):
        self.epochs = epochs
        self.model_name = model_name
        self.__set_real_min_max(self.data_ret_dir)
        self.stock_data = processed_stock(path=self.data_ret_dir, seq_len=self.seq_len)
        self.gan_args = ModelParameters(batch_size=self.batch_size,
                                       lr=self.learning_rate,
                                       noise_dim=self.noise_dim,
                                       layers_dim=self.dim)
        self.synth = TimeGAN(model_parameters=self.gan_args, 
                             hidden_dim=self.hidden_dim, 
                             seq_len=self.seq_len, 
                             n_seq=self.n_seq, 
                             gamma=self.gamma)
        self.synth.train(self.stock_data, train_steps=self.epochs) 
        if save:
            self.synth.save(f'SavedPKLs/synthesizer_{self.model_name}_{self.epochs}.pkl')
        
    def generate_synthetic(self, synthetic_size):
        self.synthetic_size = synthetic_size
        self.synthetic_data = self.synth.sample(self.synthetic_size)
        
        self.synthetic_data = self.synthetic_data[np.random.choice(self.synthetic_data.shape[0],
                                                                   size=self.synthetic_size, 
                                                                   replace=False),:]
        
        return self.synthetic_data
    
    def rescaled_cum_return(self, synthetic_data):
        self.rscld_cum_return = []
        temp = []
        i = 0
        while i < len(synthetic_data):
            temp.append(np.cumprod(self.__e_func(
                                   self.__reverse_min_max_scaler(
                                   synthetic_data[i][:,self.column_of_interest]))))
            i += 1

        self.rscld_cum_return = [np.insert(x,0,1) for x in temp]
            
        return self.rscld_cum_return

    def plot_return(self, num_paths=10000, synthetic_data=[], rescaled=True, alpha=0.5):
        if len(synthetic_data) == 0:
            synthetic_data = self.generate_synthetic(num_paths)
        else:
            num_paths = len(synthetic_data) 
        i = 0
        while i < num_paths:
            if rescaled:
                plt.plot(self.__reverse_min_max_scaler(synthetic_data[i][:,self.column_of_interest]),alpha=alpha)
            if not rescaled:
                plt.plot(synthetic_data[i][:,self.column_of_interest], alpha=alpha)
            i += 1
        plt.xlabel("Handelstage")
        plt.ylabel("Tägliche Rendite")
        plt.show()
      
    def plot_cum_return(self, num_paths=10000, data=[], alpha=0.5, bins=50):
        if len(data) == 0:
            data = self.rescaled_cum_return(self.generate_synthetic(num_paths))
        else:
            num_paths = len(data)
        
        ST = [x[-1] for x in data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'width_ratios': [3, 1]})
        mean, std = norm.fit(ST) 
        fig.suptitle(f"Endwert nach {self.seq_len} Handelstagen: μ = {round(mean,4)} σ = {round(std,4)}")

        #ax1
        i = 0
        while i < len(data):
          ax1.plot(data[i], alpha=alpha)
          i += 1
        ax1.set_xlabel("Handelstage")
        ax1.set_ylabel("Kumulierte Endwerte")
        
        #ax2
        ax2.hist(ST, bins=bins, density=True, alpha=alpha, color='b',orientation="horizontal")
        x = np.linspace(min(ST), max(ST), 100)
        ax2.plot(norm.pdf(x, mean, std), x, 'r', linewidth=2)
        distribution_info = (ax2.axhline(y=mean, color='k', linestyle='-'),
                             ax2.axhline(y=mean+std, color='k', linestyle=':'),
                             ax2.axhline(y=mean-std, color='k', linestyle=':'))
        ax2.legend(distribution_info, ["μ", "μ ± σ"])
        ax2.set_xlabel("Häufgikeit")
        
        plt.show()
    
    def plot_real_vs_synthetic_hist(self, density=True, data_synthetic = [], sample_length=10000):
        data_real = self.__slice_df(self.data_abs_dir)
        if len(data_synthetic) == 0:
            if density:
                data_synthetic = self.rescaled_cum_return(self.generate_synthetic(sample_length))
            else:
                data_synthetic = self.rescaled_cum_return(self.generate_synthetic(len(data_real)))
        else:
            sample_length = len(data_synthetic)

        ST_real = [x.iloc[-1] for x in data_real]

        ST_synthetic = [x[-1] for x in data_synthetic]

        bins = np.linspace(min(min(ST_real),min(ST_synthetic)), max(max(ST_real),max(ST_synthetic)), 50)

        mean_real = np.mean(ST_real)
        mean_synthetic = np.mean(ST_synthetic)
        std_real = np.std(ST_real)
        std_synthetic = np.std(ST_synthetic)
        wasserstein_metric = wasserstein_distance(ST_synthetic, ST_real)

        plt.hist(ST_real, bins, density=density, alpha = 0.5, label='real', color="red")
        plt.hist(ST_synthetic, bins, density=density, alpha = 0.5, label='synthetisch',color="blue")

        plt.axvline(x=mean_real, alpha = 0.5, linestyle = "-.", color='red', label="μ real")
        plt.axvline(x=mean_synthetic, alpha = 0.4, linestyle = "-", color='blue', label="μ synthetisch")

        plt.axvline(x=mean_real + std_real, alpha = 0.5, linestyle = "--", color='red', label="μ ± σ real")
        plt.axvline(x=mean_real - std_real, alpha = 0.5, linestyle = "--", color='red')

        plt.axvline(x=mean_synthetic + std_synthetic, alpha = 0.5, linestyle = ":", color='blue', label="μ ± σ synthetisch")
        plt.axvline(x=mean_synthetic - std_synthetic, alpha = 0.5, linestyle = ":", color='blue')
        
        plt.title(f"Endwert nach {self.seq_len} Handelstagen:\n" + 
                  f"Real: μ  = {round(mean_real,4)} σ = {round(std_real,4)}\n" +
                  f"Synthetisch:  μ  = {round(mean_synthetic,4)} σ = {round(std_synthetic,4)}\n" +
				  f"Wasserstein-Metrik = {round(wasserstein_metric,4)}")
        
        plt.legend()
        
        plt.xlabel("Endwerte")
        plt.ylabel("Häufigkeit")

        plt.show()
    
    def plot_t_SNE(self, sample_length, alpha=0.2):
        data_real = self.__slice_df(self.data_abs_dir)
        data_synthetic = self.rescaled_cum_return(self.generate_synthetic(len(data_real)))
            
        df1 = pd.DataFrame(data=random.sample(data_real,sample_length))
        df1["Daten"] = "real"
        
        df2 = pd.DataFrame(data=random.sample(data_synthetic,sample_length))
        df2["Daten"] = "synthetisch"
        
        df3 = pd.concat([df1,df2])
        df3.reset_index(drop=True, inplace=True)
        
        df3_numeric = df3.drop(["Daten"],axis=1)
        
        tsne = TSNE(learning_rate=50)
        
        features = tsne.fit_transform(df3_numeric)
        
        df3["X"] = features[:,0]
        df3["Y"] = features[:,1]
    
        color_dict = dict({"real":"red","synthetisch":"blue"})
        sns.scatterplot(x="X", y="Y", hue="Daten", palette=color_dict, data=df3, alpha=alpha).set(title="t-SNE")
    
    def __reverse_min_max_scaler(self, scaled):
        rescaled_value = (scaled*(self.real_max - self.real_min)) + self.real_min
        return rescaled_value
    
    def __e(self,x):
        result = math.exp(x)
        return result
    
    def __set_real_min_max(self, data_dir):
        self.real_max = pd.read_csv(data_dir)[self.columns[self.column_of_interest]].max()
        self.real_min = pd.read_csv(data_dir)[self.columns[self.column_of_interest]].min()
        
    def __load_real_data(self, dir):
        self.real_data = pd.read_csv(dir)
        return self.real_data
    
    def __slice_df(self, dir, scaled=True):
        df = self.__load_real_data(dir)
        data = []
        i = 0
        while i < len(df) - self.seq_len:
            lower = 0 + i
            upper = 0 + i + self.seq_len
            if scaled:
                data.append((df[self.columns[self.column_of_interest]][lower:upper] / 
                             df[self.columns[self.column_of_interest]][lower])
                            .reset_index(drop=True))
            else:
                data.append((df[lower:upper])
                            .reset_index(drop=True))
            i += 1
        return data