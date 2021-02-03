from Dataset import * 
import random
import numpy as np
from penalty_decomposition import *
from quadratic_test_problem import *
from inexact_penalty_decomposition import * 
from regressione_lineare import * 

#alcune inits
numVar = 12
constraint = 5
filename = "v" + str(numVar) + "c"+ str(constraint) + datetime.now().strftime("%d-%m-%Y_(%H)(%M)(%S)")
func = QuadraticTestProblem(N=numVar, n = 2, s = 4, m = 20, saveProblem = True, filename = filename+"_problem")




def local_search(x0 = None, exact = False):
    if exact: 
        pd = PenaltyDecomposition(func, tau_zero=1, x_0=x0, gamma=1.1, max_iterations=9999999999999999, save=False, l0_constraint=constraint)
    else:
        pd = InexactPenaltyDecomposition(func, tau_zero=1, x_0=x0, gamma=1.1, max_iterations=9999999999999999, save=False, l0_constraint=constraint)
    pd.start()
    fval = pd.resultVal
    fargmin = pd.resultPoint
    #print(fargmin)
    return fval, fargmin

def differential_evolution(F = 0.5, CR = 1, P = 50, n = None, bound = 100, numero_epoche = 1, exact = False):
    assert (F > 0 and F < 2)
    assert (CR >= 0 and CR <= 1)
    assert P > 0 # P is the population size
    assert n is not None
    print(f"Bound: {bound} \n Constraint: {constraint}")
    population = ((np.random.rand(n, P) - 0.5)*2)*bound # default è 1, le componenti possono andare solo tra -1 e 1
    # for each element of the population do a local search
    population_fvalues = [None] * P
    
    print("Initialization ..")
    for i in range(0, P):
        population_fvalues[i], _  = local_search(x0 = population[:,i].reshape(n, 1), exact=exact)
        print(f"Initialized {i} of {P-1}")

    print(population_fvalues)


    ordered_P = np.arange(P)
    for epoca in range(0, numero_epoche):
        print(f"Epoca {epoca}")
        for i in range(0, P):
            i_ = random.randint(0, n-1) #perchè i vettori iniziano da 0
            keys = generatek012(i, ordered_P)
            trial = (population[:,keys[0]] + F*(population[:, keys[1]] - population[:, keys[2]])).reshape((n,1))
            for j in range(0, n):
                if j != i_:
                    if random.uniform(0,1) < CR: 
                        trial[j] = population[j][i] # TODO check correctness
            
            trial_fvalue, trial_argmin = local_search(x0 = trial, exact=exact)
            if trial_fvalue < population_fvalues[i]:
                # a = population[:,1]
                population[:,i] = trial_argmin.reshape(n,)
                population_fvalues[i] = trial_fvalue
                
            print(f"Epoca {epoca}, individuo {i}. Fvalues: \n {population_fvalues}")
    print(population)
    return population_fvalues



def generatek012(i, pvec):
    np.random.shuffle(pvec)
    temp = pvec[0:3]
    while i in temp:
        #print(temp)
        np.random.shuffle(pvec)
        temp = pvec[0:3]
    return temp

def prepare_linear_regression(dataset_name = "housing"):
    data = Dataset(name=dataset_name, directory="./datasets/")
    X, Y = data.get_dataset()
    Y = np.array([Y])
    Y = Y.transpose()
    print("Shape X " + str(X.shape)) 
    print("Shape Y " + str(Y.shape)) 
    fun = RegressioneLineare(X, Y)
    #numVar = fun.n
    constraint = int(fun.n/2)
    return (fun, fun.n, constraint)


### main
# n = function.numberOfVariables()
#func, numVar, constraint = prepare_linear_regression("housing")

differential_evolution(n = numVar, P=10, CR = 0.5, numero_epoche=10, exact=False, bound = 10) #ATTENZIONE AL BOUND, se test "quadratico" limitare!