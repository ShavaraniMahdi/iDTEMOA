# FIXME: Document these functions
import numpy as np
from pygmo import fast_non_dominated_sorting, pareto_dominance
from sklearn.preprocessing import MinMaxScaler
def dominated_by_pop(f, pop_f):
    """Return True if f is dominated by pop_f"""
    for fi in pop_f:
        if pareto_dominance(fi, f):
            return True
    return False

def select_best_N_mo(f, N, pref_fun):
    # FIXME: Add corner cases: https://github.com/esa/pagmo2/blob/b97d8885fa53f0bfe74a8bb3a33809fa1a460806/src/utils/multi_objective.cpp#L346
    ndf, _, _, ndr = fast_non_dominated_sorting(f)
    
    #This part is added according to BCEMOA
    # idx=np.empty(0, dtype=int)
    # rank=0
    # while len(idx) + len(ndf[rank])<=N:
    #     idx=np.append(idx, ndf[rank][:N])
    #     rank+=1
    
    # if len(idx)<N:
    #     pref=pref_fun(f[ndf[rank]])
    #     idx= np.append(idx, ndf[rank][pref.argsort()]).astype(int)
  ############Original is here:                                                
    
    # Insert whole non dominated fronts until reaching N
    pref = pref_fun(f)
    # FIXME: This is probably faster
    idx = np.lexsort((pref,ndr)) # Sort by ndr, then by pref
    return idx[:N]

def tournament_selection(idx1, idx2, non_domination_rank, crowding_d):
    if non_domination_rank[idx1] < non_domination_rank[idx2]: return idx1
    if non_domination_rank[idx1] > non_domination_rank[idx2]: return idx2
    if crowding_d[idx1] > crowding_d[idx2]: return idx1
    if crowding_d[idx1] < crowding_d[idx2]: return idx2
    # np.random.random returns values between [0, 1), so [0, 0.5) is half
    # of the interval.
    return (idx1 if np.random.rand() < 0.5 else idx2)

def polynomial_mutation(problem, parent, prob_m, eta_m):    
    # Decision vector dimensions
    D = problem.get_nx()
    Dc = problem.get_ncx()
    # Problem bounds
    lb, ub = problem.get_bounds()

    # This creates a copy
    child = np.array(parent)
    
    do_mutation = np.random.rand(D) < prob_m
    eta_m += 1.
    # This implements the real polinomial mutation and applies it to the
    # non integer part of the decision vector
    for j in range(Dc): 
        if (do_mutation[j] and lb[j] != ub[j]):
            y = child[j]
            yl = lb[j]
            yu = ub[j]
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rnd = np.random.rand()
            mut_pow = 1. / eta_m
            if rnd < 0.5:
                xy = 1. - delta1
                val = 2. * rnd + (1. - 2. * rnd) * pow(xy, eta_m)
                deltaq = pow(val, mut_pow) - 1.
            else:
                xy = 1. - delta2
                val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * pow(xy, eta_m)
                deltaq = 1. - (pow(val, mut_pow))
                
            y = y + deltaq * (yu - yl)
            child[j] = np.clip(y, yl, yu)
        
    # This implements the integer mutation for an individual
    for j in range(Dc, D): 
        if do_mutation[j]:
            # We need to draw a random integer in [lb, ub], so we cannot use randint().
            child[j] = np.random.random_integers(lb[j], ub[j])
    return child

def sbx_beta(y1, y2, yl, yu, r, eta_c):
    beta = 1. + (2. * (yu - yl) / (y2 - y1))
    alpha = 2. - np.power(beta, -eta_c)
    if (r < (1. / alpha)):
        betaq = np.power(r * alpha, 1. / eta_c)
    else:
        betaq = np.power(1. / (2. - r * alpha), 1. / eta_c)

    return betaq

def sbx_crossover(problem, parent1, parent2, cr, eta_c):
    # Initialize the child decision vectors
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() >= cr:
        # don't do crossover
        return child1, child2

    # Decision vector dimensions
    D = problem.get_nx()
    Dc = problem.get_ncx()
    # Problem bounds
    lb, ub = problem.get_bounds()

    eta_c += 1.
    
    # This implements a Simulated Binary Crossover SBX and applies it to
    # the non integer part of the decision vector
    equal = np.abs(parent1 - parent2) > 1e-14
    y1_a =  np.minimum(parent1, parent2)
    y2_a =  np.maximum(parent1, parent2)
    # FIXME: This could be vectorized more
    for i in range(Dc):
        if np.random.rand() < 0.5 and equal[i] and lb[i] != ub[i]:
            y1 = y1_a[i]
            y2 = y2_a[i]
            yl = lb[i]
            yu = ub[i]
            rand01 = np.random.rand()

            betaq = sbx_beta(y1, y2, yl, y1, rand01, eta_c)
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                
            betaq = sbx_beta(y1, y2, y2, yu, rand01, eta_c)
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            c1 = np.clip(c1, yl, yu)
            c2 = np.clip(c2, yl, yu)

            if (np.random.rand() < .5):
                c1, c2 = c2, c1
                    
            child1[i] = c1
            child2[i] = c2
                
    # This implements two-point binary crossover and applies it to the integer part of the chromosome
    if D - Dc > 0:
        site1, site2 = np.sort(np.random.randint(Dc, D, size=2))
        child1[site1:site2] = parent2[site1:site2]
        child2[site1:site2] = parent1[site1:site2]
    return child1, child2

def problem_ideal_nadir(prob):
    name=prob.get_name()
    m=prob.get_nobj()
    n=prob.get_nx()
    k=n-m+1
    if name=='DTLZ1':
        ideal=np.zeros(m)
        nadir=np.full(m,0.500*(1+(100*(k*(2.25000)))))
    elif name=='DTLZ2':
        g=k*0.25000
        ideal=np.zeros(m)
        nadir=np.full(m,(1+g))
    elif name=='DTLZ7':
        x=np.linspace(0,1,10000000)
        f=(x*(1+np.sin(3*np.pi*x)))
        f_min=f.min()
        f_max=f.max()
        g_min=1
        g_max=10
        ideal=np.append(np.zeros(m-1),(1+g_min)*(m-(m-1)*f_max/(1+g_min)))
        nadir=np.append(np.ones(m-1), (1+g_max)*(m-(m-1)*f_min/(1+g_max)))
    else:
        pop = pg.population(prob, size = 10000, seed=12345)
        nadir=pg.nadir(pop.get_f())
        ideal=pg.ideal(pop.get_f())
    return ideal, nadir