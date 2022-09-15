#importing libraries
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time
from sklearn import svm, tree
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from copy import copy,deepcopy
from csv import writer
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
# MANUEL: This should not be necessary.
from problems import as_dummy
#import decision_tree as dt
from ea_operators import tournament_selection, polynomial_mutation, \
    sbx_crossover, select_best_N_mo

'''
*

* @param[in] cr Crossover probability.
* @param[in] eta_c Distribution index for crossover.
* @param[in] m Mutation probability.
* @param[in] eta_m Distribution index for mutation.
* @param seed seed used by the internal random number generator (default is random)
* @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
* [1,100[ or \p eta_m is not in [1,100[.
* @param gen1: number of nsga2 generations before the first interaction of DTEMOA
* @param geni: number of generations between two consecutive interactions
* @param sampleSize: number of solutions presented to the DM for pairwise comparison
* @param detection:  determines if the detection of hidden objectives and consequently the update of objs are active or not                                   
*/'''    
#Main DTEMOA Class
#Defining DTEMOA class based on nsga2 class in Pygmo
class dtemoa():
    def __init__ (self, dm, total_gen=500, gen = 100, cr = 0.95, eta_c = 10., m = 0.01, eta_m = 50.,
                  seed = 564654561, geni=20, interactions=10, sampleSize=5, verbose = 0):
        np.random.seed(seed)
        self.gen=gen
        self.nsga2 = pg.algorithm(pg.nsga2(gen=gen, cr=cr, eta_c=eta_c, m=m, eta_m=eta_m, seed=seed))
        self.geni = geni
        self.interactions=interactions
        self.mdm = dm
        self.m_cr = cr
        self.m_m = m
        self.m_eta_c = eta_c
        self.m_eta_m = eta_m
        self.sampleSize = sampleSize
        self.m_log = []
        self.total_gen=total_gen
        self.verbose = verbose
    def get_name(self):
        return 'DTEMOA'
    # Overloading evolve function of nsga2   
    def evolve(self,pop):
        pop = self.nsga2.evolve(pop)
        pop = self.evolvei(pop)
        return pop

        
    def evolvei(self, pop):
        #keeping the records of best values through interactions
        vf_inter=[self.mdm.value(pop.get_f()[0],1)]
        N = len(pop)
        if N < self.sampleSize:
            print(f"Warning: Sample size={self.sampleSize} was larger than N={N}, adjusting it")
            self.sampleSize = N
        self.total_gen-=self.gen#gen iterations have been performed in nsga2 prior to evolvei       
        prob = pop.problem
       
        # A list of fitness vectors for training
        training_set=[]
        training_x=[]
        pairwise=[]
        rank_pref = np.empty(0)

        for LearningIteration in range(self.interactions):
            #####################################################
            # Interaction and Learning  
            #####################################################
            #if first dolution dominates second the label is 1, otherwise -1
            training_set, training_x, rank_pref, pairwise= self.get_preferences(pop,training_set,training_x, rank_pref, pairwise)
            #training_set, rank_pref, pairwise= self.get_preferences(pop,training_set, rank_pref, pairwise)
            if self.mdm.mode != 1: # rank by svm;
                p=np.asarray(pairwise)
                p[:,-1]=p[:,-1].astype(int)
                # self.tree=dt.build_tree(p)#pairwise
                
                #Alternative
                if len(training_set)<=10:
                    cv=3
                else:
                    cv=5
                param_grid = {'splitter':['best', 'random'], 'max_depth':list(np.arange(10,120)), 'min_samples_leaf':[1,2,3]}#,5
                clf=GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=cv)
                self.clf=clf.fit(p[:,:-1], p[:,-1])
                print(clf.best_score_)
                pref_fun=self.score
            else:# when the mode is 1 no learning is done
                pref_fun = self.mdm.value_array
              
            ######################################################
            # dtemoa iterations: geni iterations after each interaction  
            ######################################################
            if LearningIteration == self.interactions-1:
                self.geni = self.total_gen - (self.interactions - 1) * self.geni
            # The NSGA2 loop with preference replacing the crowding distance
            for gens in range(self.geni):
                f = pop.get_f()
                pop_pref = pref_fun(f)   
                _, _, _, ndr = pg.fast_non_dominated_sorting(f)
                shuffle1 = np.random.permutation(N)
                shuffle2 = np.random.permutation(N)
                x = pop.get_x()
                # We make a shallow copy to not change the original pop.
                pop_new = deepcopy(pop)
                for i in range(0, N, 4):
                    child1, child2 = self.generate_two_children(prob, shuffle1, i, x, ndr, -pop_pref)
                    # we use prob to evaluate the fitness so that its feval
                    # counter is correctly updated
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))
                    
                    child1, child2 = self.generate_two_children(prob, shuffle2, i, x, ndr, -pop_pref)
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))

        
                # Selecting Best N individuals
                f = pop_new.get_f()
                best_idx = select_best_N_mo(f, N, pref_fun)
                                
                assert len(best_idx) == N
                x = pop_new.get_x()[best_idx]
                f = f[best_idx]
                for i in range(len(pop)):
                    pop.set_xf(i, x[i], f[i])
                if self.verbose>1:
                    print(f"best vf: {self.mdm.value(pop.get_f()[0])}")
            if self.verbose:
                plt.clf()
                # fig = plt.figure(figsize = (10, 7)) 
                # ax =plt.axes(projection ="3d") 
                # ax.scatter3D(scaled_af[:,0], scaled_af[:,1], scaled_af[:,2])
                plt.scatter(pop.get_f()[:,0], pop.get_f()[:,1])
                plt.xlim(0.0, None)
                plt.ylim(0.0, None)
                plt.pause(0.000001)
            vf_inter.append(self.mdm.value(pop.get_f()[0],1))
        # if self.interactions>=6:
            # fig, ax = plt.subplots()
            # plt.ylim(min(vf_inter)-abs(min(vf_inter)*0.05),max(vf_inter)*(1.05))
            # ax.plot(np.arange(len(vf_inter)),vf_inter)
            # ax.set_title("vf vs. number of interactions: DTEMOA, Problem: {}, fDim: {}, mode: {}".format(prob.get_name(), prob.get_nobj(), self.mdm.mode))
            # ax.set(xlabel='interactions',ylabel='vf')
            # pdf_filename = "Trend. DTEMOA, Problem {}, fDim {}, mode {}, sigma {}, gamma {}, q {}".format(prob.get_name(), prob.get_nobj(), self.mdm.mode, self.mdm.sigma, self.mdm.gamma, self.mdm.q)+".pdf"
            # fig.savefig(pdf_filename)
            # plt.show()
            
            
            # vf_inter.extend(['DTEMOA', prob.get_name(),prob.get_nobj(), self.mdm.mode ,self.mdm.sigma, self.mdm.gamma, self.mdm.q])
            # file = open('trend.csv','a', newline='')
            # csv_writer = writer(file)
            # csv_writer.writerow(vf_inter)
            # file.close()
        print('evolve of DTEMOA ended')
        return self.last_interaction(pop, training_x, training_set, rank_pref)
    
    
    def get_preferences(self, pop, training_set,training_x, rank_pref, pairwise):
        prob=pop.problem
        # This function assumes that pop is already sorted by non-dominated
        # sorting breaking ties with the predicted utility function.

        remaining = self.sampleSize
        # F=Filter(self.sampleSize, pop.get_f()) 
        F=np.random.choice(len(pop), self.sampleSize, replace= False)
        
        # MANUEL: Remove duplicates is counter-productive when you may have
        # hidden objectives because the DM behaving different for solutions
        # that look the same is helpful to detect hidden objectives.
        # for i in range(len(pop)):
        for i in F:
            f = pop.get_f()[i]
            x = pop.get_x()[i]
            training_set.append(f)
            training_x.append(x)

        start = len(rank_pref)
        # This is where the interaction with the DM occurs.
        tmp_rank_pref = self.mdm.setRankingPreferences(training_set[start:])
        rank_pref=np.append(rank_pref, tmp_rank_pref)
        # training_data=[]
        for i in range(start, len(training_set)):
            for j in range(i+1,len(training_set)):
                #sign can be replaced by rounded integers
                diff=(np.asarray(training_set[i])-np.asarray(training_set[j])).tolist()
                #each example contains differences between 2 objective vectors and if the first one is pereferred to the second one as label indicated by (-1 , 1).
                diff.append(np.sign(rank_pref[j]-rank_pref[i]))
                pairwise.append(diff)
        return training_set, training_x, rank_pref, pairwise

    
    def crossover(self, problem, parent1, parent2):
        return sbx_crossover(problem, parent1, parent2, self.m_cr, self.m_eta_c)
      
    def mutate(self, problem, child):
        return polynomial_mutation(problem, child, self.m_m, self.m_eta_m)
                          
    def generate_two_children(self, problem, shuffle, i, X, ndr, pop_pref):
        parent1_idx = tournament_selection(shuffle[i], shuffle[i + 1], ndr, pop_pref) 
        parent2_idx = tournament_selection(shuffle[i + 2], shuffle[i + 3], ndr, pop_pref)
        parent1 = X[parent1_idx]
        parent2 = X[parent2_idx]
        child1, child2 = self.crossover(problem, parent1, parent2)
        child1 = self.mutate(problem, child1)
        child2 = self.mutate(problem, child2)
        return child1, child2

    def get_log(self):
        return self.m_log
    
    def last_interaction(self, pop, training_x, training_set, rank_pref):
        best_vf=self.mdm.value(pop.get_f()[0],1)
        for i in  rank_pref.argsort()[:self.sampleSize]:
            if self.mdm.value(training_set[i],1)<best_vf:
                pop.set_x(0,training_x[i])
        return pop
    def score(self,f):
        #Finding the scores for each member of the population
        #f: objective vectors of the population
        #tree: the tree build on the training_set
        scors=[]
        for i in range(len(f)):
            temp=0
            for j in range(len(f)):
                if i==j:
                    continue
                diff=f[i]-f[j]
                t=self.clf.predict_proba(diff.reshape(1,-1))
                if 1 in t:
                    temp+=t[0][1]
            scors.append(temp)
        return -np.asarray(scors)    


# def make_tree(training_set, rank_pref, pairwise,  verbose=False):
#     #Constructing training data for 
#     training_data=[]
#     for i in range(len(training_set)):
#         for j in range(i+1,len(training_set)):
#             #sign can be replaced by rounded integers
#             diff=np.round(np.asarray(training_set[i])-np.asarray(training_set[j]),2).tolist()
#             #each example contains differences between 2 objective vectors and if the first one is pereferred to the second one as label indicated by (-1 , 1).
#             diff.append(np.sign(rank_pref[j]-rank_pref[i]))
#             training_data.append(diff)
            
#     return dt.build_tree(pairwise)

          
def Filter(P, af):
   
    # 3) Calculate rectilinear distances dkl between each pair of
    # individuals k, l ∈ F1.
    dist = distance.pdist(af, metric='cityblock')
    # Transform condensed to square distance matrix
    dist = distance.squareform(dist)
            
    # 4) Initialize the filtered list F2 by moving a pair of
    # individuals (yk , yl ) = argmax(yu ,yv ) ∈ F1 (d_uv) from af to
    # F2. That is, choose the pair of individuals that are farthest to each
    # other in rectilinear distance and move them from af to F2 .
    k, l = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    # We keep two lists of indexes, in_F1 are indexes of af that are still in af and in_F2 those moved to F2 
    in_F1 = list(range(len(af)))
    in_F1.pop(l)
    in_F1.pop(k)
    F2 = [k, l]
            
    # 5) Fill F2 until its size is equal to P, each time moving solution
    # y_k = argmax_{y_u ∈ F1} min (y_v ∈ F2 (d_uv ) to F2. That is, move
    # the individual y_k in F1 which is at maximum distance to its closest
    # individual in F2.
    while len(F2) < P:
        # Select rows in_F1 and columns in_F2, calculate the minimum per
        # row, then maximum of the minimums
        min_dist = np.min(dist[np.ix_(in_F1, F2)], axis=1)
        assert len(min_dist) == len(in_F1)
        idx = np.argmax(min_dist)
        F2.append(in_F1.pop(idx))
    # F2 = [ F1[i] for i in F2 ]
    
    
    return F2
