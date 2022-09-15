import numpy as np
from scipy.stats.stats import pearsonr
from ea_operators import problem_ideal_nadir
# =============================================================================
# Defining MachineDM:Constructs a Machine Decision Maker
# =============================================================================
class  machineDM:
    '''
    *

    * @param seed seed used by the internal random number generator (default is random)
    * @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
    * [1,100[ or \p eta_m is not in [1,100[.
    * @param pref: the reference to the preference/utility function class
    * @param prob: reference to the problem at hand
    * @param sigma: is standard deviation, sigma^2 is variance. used for adding noise the evaluation of the DM
    * @param gamma: controls dependency among objectives. used for combining the values of the objs
    * @param q: number of ommited objectives
    * @param tau: as described in stewart value function, the indifference obj value that the ommited objs are set to
    * @param Mode options:
        1. Learning: off , biases: off (golden stanndard)
        2. Learning: on, biases: off 
        3. Learning:on, biases:on
    */'''
    
    def __init__(self, prob, pref, mode, gamma=0, sigma=0, q=0, tau=0.5 , scaling=False, ideal=None, nadir=None, verbosity=0, seed = 564654561):
        #problem &prob, value_function &pref, unsigned mode, double gamma, double sigma, double int q,
        # FIXME: Add checking of input parameters
        np.random.seed(seed)
        self.prob = prob
        self.pref = pref # This is a value function
        self.verbosity=verbosity
        assert(mode >= 1 and mode <= 4)
        self.mode = mode # mode is either 1: gold standard, 2: with interaction, 3: with interaction and biases 
        self.gamma = gamma
        self.sigma = sigma
        self.q = q
        if type(tau)==float:#when tau is not passed to the function
            self.tau = np.full(prob.get_nobj(),tau)
        else:
            self.tau=tau
        if scaling:
            _, self.f_max=problem_ideal_nadir(prob)
        
        m = self.prob.get_nobj()
        # MANUEL: Why do you need a copy of weights here?
        if hasattr(self.pref,'weights'):
            weights = self.pref.weights
        else:
            weights = np.full(m, 1 / m)
        # q is the number that are omitted, idx is the indice of objectives that are preserved
        if q>0:
            idx = np.random.choice(m, m-q, p = weights, replace=False)
        else:
            idx=np.arange(m)
        omitted = np.ones(m, dtype=bool)
        omitted[idx]= False
        assert np.sum(omitted) == q
        self.omitted = omitted

        self.ideal=ideal
        self.nadir=nadir
        
        c = np.arange(m)
        if gamma > 0:
            if m-q>1:
                temp=idx
                while np.array_equal(idx,temp):
                    idx=np.random.permutation(idx)
                c[temp] = c[idx]
            else:
                print("The problem is now turned to a single-objective")
        self.c = c
        
        self.scaling=scaling


    def original_fitness(self, x):
        assert len(x) == self.prob.get_nx()
        return self.prob.original_fitness(x)
    
    def corr_cal(self,f):
        if self.scaling:
            f=(f-self.ideal)/(self.nadir-self.ideal)
        true_rank=np.argsort(np.array([self.pref.value(fi) for fi in f]))
        biased_rank=np.argsort(np.array([self.dm_evaluate(fi) for fi in f]))
        return pearsonr(true_rank,biased_rank)[0]

    

    def value(self, obj, mode=None):
        if mode==None:
            mode=self.mode
        
        """This gives the utility value of obj vector"""
        #The scaling should be done here and before applying biases, since the bounds may change due to mixing
        obj_=obj*1.
        if self.scaling:
            obj_=0.95*(obj-self.ideal)/(self.nadir-self.ideal)
            obj_[obj_>0.95]=0.95+0.05*obj[obj_>0.95]/self.f_max[obj_>0.95]
            obj_[obj_<0]=0
            obj_[obj_>1]=1
        # assert np.all(obj >= 0)
        # assert np.all(obj <= 1.01)
        
        if mode == 1 or mode == 2 :
            return self.pref.value(obj_)
        else:
            return self.dm_evaluate(obj_)
        
            

    def value_array(self, f, mode=None):
        
        """This gives the utility value of a whole matrix of obj vectors"""
        return np.array([self.value(fi,mode) for fi in f])
        
    def modify_criteria(self, obj):
        zhat = np.array(obj)*1 # We want to copy it
        if self.gamma != 0:
            zhat = zhat[self.c] * self.gamma + zhat * (1.0 - self.gamma)
        zhat[self.omitted] = self.tau[self.omitted]
        return zhat

    def dm_evaluate(self, obj):
        """
          (b) the addition of a noise term, normally
          distributed with zero mean and a variance of
          sigma^2 (which will be a specified model parameter),
        """
        #M: Do we need to scale?     = this->prob.get_ub(); // M: Scaling the obj to [0,1] by deviding each obj value by its upperbound.
        #std::pair<vector_double, vector_double> bounds = this->prob.get_bounds();
        z_mod = self.modify_criteria(obj)
        
        # sigma is standard deviation, sigma^2 is variance.
        # This produces a value in N(0, sigma^2)
        noise = (self.sigma * np.random.randn()) if self.sigma > 0 else 0.0
        # Utilities must be between 0 and 1
        v = self.pref.value(z_mod)
        # assert v >= 0 and v <= 1
        # estim_v = np.clip(noise + v, 0, 1)
        return noise + v

    def most_preferred(self, z):
        dm_values = self.value_array(z)
        # FIXME: Utility should be maximized 
        return dm_values.argmin()
        
    def setRankingPreferences(self, z):
        """This function is used for pairwise comparison of solutions by DM"""
        #checking the correlation between biased and true rankings
        if self.verbosity:
            print(f'correlation among biased and unbiasded training set: {self.corr_cal(z)}')
            
        len_z = len(z)
        rank_pref = np.zeros(len_z)
        values = self.value_array(z)
        # if self.pref.get_name()=="stewart":
        #     values=1-values
        return values.argsort().argsort()
        # for i in range(len_z):
        #     for j in range(i+1, len_z):
        #         pref = values[i] - values[j]
        #         if pref < 0: rank_pref[j] += 1  
        #         if pref >= 0: rank_pref[i] += 1
        # return rank_pref

