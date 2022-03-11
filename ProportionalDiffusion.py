
# coding: utf-8

# In[ ]:


import scipy
from scipy import optimize
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ProportionalDiffusion(object):
    
    def __init__(self, rt=None, accuracy=None, stimulus_strength=None, required_accuracy=None):
        """ Initalizes 
        
        Parameters
        -----------
        rt: list-like (list or np.array or pd.Series; anything that pd.DataFrame understands)
            Reaction times to fit
        accuracy: list-like
            Accuracies to fit
        stimulus_strength: list-like
            Stimulus strength corresponding to rt and accuracy arguments
        required_accuracy: float
            Accuracy you want to get the corresponding stimulus strength for
        """
        
        if rt is not None and accuracy is not None and stimulus_strength is not None:
            self.data = pd.DataFrame({'rt': rt, 'accuracy': accuracy, 'stimulus_strength': stimulus_strength})

            # remove null responses
            self.data = self.data.loc[pd.notnull(self.data['rt'])]
        else:
            self.data = None
        
        self.kfit = None
        self.t0fit = None
        self.Aprimefit = None
        self.required_accuracy = required_accuracy
    
    def Pc(self, x, k, Aprime):
        """ Proportion correct """
        
        return 1/(1+np.exp(-2*Aprime*k*np.abs(x)))

    def Tt(self, x, k, Aprime, t0):
        """ Mean RT """
        
        return (Aprime/(k*x)) * np.tanh(Aprime * k * x) + t0

    def get_stimulus_strength_for_accuracy(self, accuracy, k=None, Aprime=None, lower=None, upper=None):
        """ Solve eq. 1 for x to get stimulus strength for a given accuracy level """
        
        if k is None and Aprime is None:
            k = self.kfit
            Aprime = self.Aprimefit
        
        x = np.log((1-accuracy)/accuracy) / (-2*Aprime*k)

        if lower is not None:
            if x < lower:
                x = lower
        if upper is not None:
            if x > upper:
                x = upper

        return x 

    def obj(self, pars):
        """ Objective function for fitting """
        
        k = pars[0]
        Aprime = pars[1]
        t0 = pars[2]

        observed_means = self.data.groupby('stimulus_strength').mean()
        unique_strengths = observed_means.index.values
        predicted_sems = self.data.groupby('stimulus_strength').sem()  # predicted SE of mean
        predicted_means = self.Tt(unique_strengths, k, Aprime, t0)

        unique_strengths = observed_means.index.values
        observed_means = observed_means['rt'].values
        predicted_sems = predicted_sems['rt'].values

        # Eq 3
        dev = np.divide(1, np.multiply(predicted_sems, np.sqrt(2*np.pi)))
        exponent = np.exp(-(predicted_means - observed_means)**2) / (2*predicted_sems ** 2)
        likelihood_rts = np.multiply(dev, exponent)

        # Eq 4
        n_total = self.data.groupby('stimulus_strength')['accuracy'].size().values
        n_correct = self.data.groupby('stimulus_strength')['accuracy'].sum().values
        likelihood_accs = scipy.special.comb(n_total, n_correct) * self.Pc(x=unique_strengths, Aprime=Aprime, k=k)**n_correct * (1-self.Pc(x=unique_strengths, Aprime=Aprime, k=k))**(n_total-n_correct)

        negLL = -np.sum(np.concatenate((np.log(likelihood_accs), np.log(likelihood_rts))))

        return(negLL)

    def fit(self):
        """ Fits model to provided data """
        
        print('Fitting...')
        if self.data is None:
            raise(IOError('No data provided to be fitted...'))

        bounds = [(0, 200), (0, 100), (0, 2)]  # k, Aprime, t0
        opt = scipy.optimize.differential_evolution(func=self.obj, 
                                                    bounds=bounds)
        self.kfit = opt.x[0]
        self.Aprimefit = opt.x[1]
        self.t0fit = opt.x[2]
        self.opt = opt
        
        if not opt.success:
            print('WARNING! Convergence was unsuccessful!')
        
        print('done')
        
    def simulate_diffusion(self, n=1000, a=None, v=None, t0=None, required_accuracy=None):
        """ Function to simulate the Diffusion model without z, sz, sv, st. 
        Aka EZ diffusion / proportional drift rate diffusion model.
        
        **MUCH** more efficient simulation methods are available in, e.g., HDDM; this is included here only to be able to
        simulate within psychopy and not rely on external packages (HDDM is not part of the standalone version of PsychoPy)
        
        a = boundary sep
        v = drift
        t0 = non-decision time

        Note that this simulates with bounds parametrized as [0, a] instead of [-a, a] as the prop drift rate model uses.
        Therefore, multiply Aprimefit by 2
        """
        
        if a is None and v is None and t0 is None:
            if required_accuracy is None:
                if self.required_accuracy is None:
                    raise(IOError('I need to know what the required accuracy is to determine the drift rate...'))
                else:
                    required_accuracy = self.required_accuracy
            
            a = self.Aprimefit*2
            v = self.kfit*self.get_stimulus_strength_for_accuracy(accuracy=required_accuracy, k=self.kfit, Aprime=self.Aprimefit)
            t0 = self.t0fit
        
        si=1 #scaling factor
        M = np.pi*si**2/a**2 * (np.exp(a*v/(2*si**2))+np.exp(-a*v/(2*si**2))) * 1/ (v**2/(2*si**2)+np.pi**2*si**2 / (2*a**2))
        lmb = v**2/(2*si**2) + np.pi**2*si**2/(2*a**2)
        eps=1e-15
        ou=[]
        rej=0

        while len(ou) < n:
            w=np.random.uniform(0, 1, 1) 
            u=np.random.uniform(0, 1, 1)
            FF=np.pi**2*si**4 * 1/(np.pi**2*si**4+v**2*a**2)
            sh1=1
            sh2=0
            sh3=0
            i=0
            while np.abs(sh1-sh2)>eps or np.abs(sh2-sh3)>eps:
                sh1=sh2
                sh2=sh3
                i=i+1
                sh3= sh2 + (2*i+1)*(-1)**i*(1-u)**(FF*(2*i+1)**2)
            eval = 1+(1-u)**-FF * sh3
            if w <= eval:
                ou.append(1/lmb*np.abs(np.log(1-u)))
            else:
                rej=rej+1
        p=np.exp(a*v)/(1+np.exp(a*v))
        chance=np.random.uniform(0, 1, n)
        x = (p > chance) *1
        
        return pd.DataFrame({'accuracy': x, 'rt':np.concatenate(ou)+t0})
    
    def plot(self, k=None, Aprime=None, t0=None, ignore_data=False, min_strenght=1e-3, max_strength=1e2, required_accuracy=None):
        """ Plots the stimulus strength-meanRT and stimulus strength-accuracy curves for given parameters.
        Possibly, add data points """

        if k is None:
            k = self.kfit
        if Aprime is None:
            Aprime = self.Aprimefit
        if t0 is None:
            t0 = self.t0fit
        
        if k is None or Aprime is None or t0 is None:
            raise(IOError('No parameters provided to plot...'))
        
        # model
        f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        stimStrength = np.arange(min_strenght, max_strength, 1e-2)
        ax[0].plot(stimStrength, self.Tt(stimStrength, k, Aprime, t0))
        ax[1].plot(stimStrength, self.Pc(stimStrength, k, Aprime))

        # data
        if self.data is not None and not ignore_data:
            strengths = self.data['stimulus_strength'].unique()
            for this_strength in strengths:
                mean_rt = self.data.loc[self.data['stimulus_strength']==this_strength, 'rt'].mean()
                sem = self.data.loc[self.data['stimulus_strength']==this_strength, 'rt'].sem()
                ax[0].plot(this_strength, mean_rt, '.', color='k')
                ax[0].errorbar(this_strength, mean_rt, xerr=0, yerr=sem, color='k')

                mean_acc = self.data.loc[self.data['stimulus_strength']==this_strength, 'accuracy'].mean()
                sem = self.data.loc[self.data['stimulus_strength']==this_strength, 'accuracy'].sem()
                ax[1].plot(this_strength, mean_acc, '.', color='k')
                ax[1].errorbar(this_strength, mean_acc, xerr=0, yerr=sem, color='k')

        ax[0].set_ylabel('mean RT')
        ax[1].set_xlabel('stimulus strength')
        ax[1].set_ylabel('accuracy')
        ax[0].set_xscale("log", nonposx='clip')
        title_text = 'A'': %.2f, k: %.2f, t0: %.2f' %(Aprime, k, t0)

        
        # Add required accuracy lines if provided, else go to default
        required_accuracy = self.required_accuracy if required_accuracy is None else required_accuracy

        if required_accuracy is not None:
            proposed_X = self.get_stimulus_strength_for_accuracy(required_accuracy, Aprime=Aprime, k=k)
            title_text = title_text + ', req strength: %.2f' %proposed_X

            strength = [0, proposed_X]
            meanRT = np.repeat(self.Tt(proposed_X, k=k, Aprime=Aprime, t0=t0), len(strength))
            meanRT2 = [0, self.Tt(proposed_X, k=k, Aprime=Aprime, t0=t0)]
            strength2 = np.repeat(proposed_X, len(meanRT2))
            ax[0].plot(strength, meanRT, 'g--')
            ax[0].plot(strength2, meanRT2, 'g--')

            strength = [0, proposed_X]
            meanAcc = np.repeat(self.Pc(proposed_X, k=k, Aprime=Aprime), len(strength))
            meanAcc2 = [0, self.Pc(proposed_X, k=k, Aprime=Aprime)]
            strength2 = np.repeat(proposed_X, len(meanRT2))
            ax[1].plot(strength, meanAcc, 'g--')
            ax[1].plot(strength2, meanAcc2, 'g--')

        ax[0].set_title(title_text)

        return f, ax

if __name__ == '__main__':
    ### Some examples for use
    
    # To just plot the accuracy ~ stim strength and rt ~ stim strength curves for a given set of parameters:
    get_ipython().magic(u'matplotlib inline')
    f, ax = ProportionalDiffusion().plot(k=10, Aprime=2, t0=.1)
    
    # To fit data, provide the reaction times, accuracy, and corresponding stimulus strengths.
    import hddm
    import pandas as pd
    # simulate data with k=10
    data, params = hddm.generate.gen_rand_data({'coh0': {'v':0.01, 'a':2, 't':.3, 'sz':0, 'sv':0},
                                                'coh2': {'v':2, 'a':2, 't':.3, 'sz':0, 'sv':0},
                                                'coh4': {'v':4, 'a':2, 't':.3, 'sz':0, 'sv':0},
                                                'coh8': {'v':8, 'a':2, 't':.3, 'sz':0, 'sv':0}}, size=1000)
    data['stimulus_strength'] = np.nan
    data.loc[data['condition']=='coh0', 'stimulus_strength'] = .001
    data.loc[data['condition']=='coh2', 'stimulus_strength'] = .20
    data.loc[data['condition']=='coh4', 'stimulus_strength'] = .40
    data.loc[data['condition']=='coh8', 'stimulus_strength'] = .80
    
    # and fit
    model = ProportionalDiffusion(rt=data['rt'], accuracy=data['response'], stimulus_strength=data['stimulus_strength'])
    model.fit()
    
    # Now, you can make some plots to check the fit quality
    _ = model.plot()
    
    # Now, you probably want to know what stimulus strength would generate data with a given accuracy - let's say, 80%
    stimulus_strength = model.get_stimulus_strength_for_accuracy(accuracy=.8)
    print(stimulus_strength)
    
    # Add this to the plot
    _ = model.plot(required_accuracy=.8)
    
    # As a final check - are we sure that this stimulus strength actually would generate 80% accuracy, 
    # assuming that the EZ/proportioanl diffusion model generates the data?
    data = model.simulate_diffusion(n=1000, required_accuracy=.8)
    print(data.describe())

    ## The mean accuracy should lie around required_accuracy

