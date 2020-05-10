#This file conains implementation for an genreric RKM method with adaptive b control.
#It contains a seperate implementation for explicit RK methods and DIRK methods


import numpy as np
import scipy.optimize as opt
from OrderCondition import *
import cvxpy as cp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, convex_hull_plot_2d



def solve_LP(solver,O,rhs,rkm,b_orig,u,K,dt,reduce = False,verbose_LP = False,minval = 0,maxval=np.infty, **options):
    """
    This method solves the LP Problem

    Parameters:

    solver:     a string with the solver that should be used
    O:          Order condition matrix
    rhs:        Order condition right hand side
    rkm:        RKM
    b_orig:     Original b
    reduce:     If set, the LP problem is first solved with a reduced set of constriants
    verbose_LP: prints additional messages
    minval:     Vector or scalar, is the minimum to enforce for solution. If not needed set to -inf
    maxval:     Vector or scalar, is the minimum to enforce for solution. If not needed set to inf
    options:    additional Options, are passed through to the used solver

    Returns:
    status: integer representing the status of the algorithm.
      For scipy
        0 : Optimization proceeding nominally.
        1 : Iteration limit reached.
        2 : Problem appears to be infeasible.
        3 : Problem appears to be unbounded.
        4 : Numerical difficulties encountered.
      For cvypy
        0 : Optimization proceeding nominally.
        2 : Problem appears to be infeasible.
        4 : Numerical difficulties encountered.
        5 : solver crashed
        6 : Trivial Problem
    l = Array with number of constraints
    b: found b, if solver failed b_orig

    """
    s = len(rkm.b)
    k = K.shape[0] #Number of ODEs

    l = []

    #Filter for variables with boundarys
    if np.isscalar(minval):
        minval = np.repeat(minval,k)
    if np.isscalar(maxval):
        maxval = np.repeat(maxval,k)

    i_n = minval > -np.infty
    i_p = maxval < np.infty


    i_min = np.zeros(k,bool)
    i_max = np.zeros(k,bool)
    if reduce: #We do not need it otherwise
        u_ = u +dt*K@b_orig

    cnt_iter_h = 0
    while True: #We check at the end for break conditions
        cnt_iter_h += 1
        #Reduce
        if reduce:
            i_min[i_n] = i_min[i_n] | (u_[i_n] < minval[i_n])
            i_max[i_p] = i_max[i_p] | (u_[i_p] > maxval[i_p])

        else:
            i_min = i_n
            i_max = i_p

        if not np.any(i_min) and not np.any(i_max):
            print('trivial Problem')
            return (0,0,b_orig)


        l.append(np.sum(i_min)+np.sum(i_max))

        #Solve LP-Problem

        u_min = u[i_min] #slice the u and K
        K_min = K[i_min,:]
        u_max = u[i_max]
        K_max = K[i_max,:]
        minval_min = minval[i_min]
        maxval_max = maxval[i_max]

        if solver == 'scipy_ip' or solver == 'scipy_sim':
            if solver == 'scipy_ip':
                method = 'interior-point'
            elif solver == 'scipy_sim':
                method = 'revised simplex'

            A_eq = np.concatenate((O,-O),axis = 1)
            b_eq = rhs - O@b_orig
            bounds = (0, None)
            e = np.ones(2*s)

            A_ub_min = np.concatenate((-K_min,K_min),axis = 1)
            A_ub_max = np.concatenate((K_max,-K_max),axis = 1)
            A_ub = np.concatenate((A_ub_min,A_ub_max),axis = 0)
            #print('A_ub',A_ub)
            b_ub_1 = 1/dt*(u_min-minval_min)+K_min@b_orig
            b_ub_2 = 1/dt*(maxval_max-u_max)-K_max@b_orig
            b_ub = np.concatenate((b_ub_1,b_ub_2),axis = 0)
            #print('b_ub',b_ub)

            try:
                res = linprog(e, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method,
                              options = options)
            except:
                if verbose_LP: print('Solver crashed')
                return (5,l,None)

            if res.success:
                b = b_orig+res.x[:s]-res.x[s:]
                status = res.status
            elif res.status == 4:
                if verbose_LP: print('Numerical difficulties, giving it a try')
                b = b_orig+res.x[:s]-res.x[s:]
                status = res.status
            else:
                if verbose_LP: print('solver did not find solution, stauts:',res.status)
                status = res.status
                if verbose_LP: print(status)
                return (5,l,None)


        else:
            ap_op =cp.Variable(s)
            an_op =cp.Variable(s)
            e = np.ones(s) #vector for goal Function, just generates the 1-Norm of b


            #Using different implementations to aviod empyt matricies
            if K_max.shape[0]>0 and K_min.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [O@(ap_op-an_op+b_orig)==rhs,
                    u_min+dt*K_min@(ap_op-an_op+b_orig)>=minval_min,
                    u_max+dt*K_max@(ap_op-an_op+b_orig)<=maxval_max,ap_op>=0,an_op>=0])
            elif K_min.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [O@(ap_op-an_op+b_orig)==rhs,
                    u_min+dt*K_min@(ap_op-an_op+b_orig)>=minval_min,ap_op>=0,an_op>=0])
            elif K_max.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [O@(ap_op-an_op+b_orig)==rhs,
                    u_max+dt*K_max@(ap_op-an_op+b_orig)<=maxval_max,ap_op>=0,an_op>=0])
            else:
                raise ValueError
                #should alredy be detected as trivial Problem

            try:
                prob.solve(solver=solver,**options)
                b = ap_op.value - an_op.value + b_orig
            except:
                if verbose_LP: print('Solver crashed')
                return (5,l,None,0)

            if prob.status == cp.OPTIMAL:
                status = 0
            elif prob.status == cp.OPTIMAL_INACCURATE:
                if verbose_LP: print(prob.status)
                status = 4
            else:
                status = 5


        #Test for break conditions
        if np.all(i_min == i_n) and np.all(i_max == i_p): #reached whole set
            break
        u_ = u +dt*K@b
        if np.all(np.greater_equal(i_min,u_<minval)) and np.all(np.greater_equal(i_max,u_>maxval)):
            #if there are no new negative values appearing
            break

    return (status,l,b,cnt_iter_h)

def Eqsys_line(B):
    N = np.array([[-1.,1.]])
    a = np.array([[np.min(B)],[-np.max(B)]])
    if a[0]+a[1] == 0:
        print('only single Point, not able to compute convex hull')
        raise ValueError
    return N,a

def Eqsys_chull(B):
    """
    Function to compute the equarionsystem defining a c_hull of the points b_1,...,b_l

    Parameters:
    B = [b_1,...,b_l] Points that span the chull

    Returns:
    Two sets of constriants
        N_hull^T x     + a_hull    <= 0
        N_subspace^T x + a_subspace = 0

    N_hull = [n_1,...,n_k] Matrix with normal vectors of the inequality constraints
    a_hull = [a_r,...,a_k] Vector containig the offsets of the inequality constraints
    N_subspace = [n_k+1,...,n_k+(r-s)] Matrix with normal vectors of the equality constraints
    a_subspace = [n_k+1,...,n_k+(r-s)] Vector containig the offsets of the equality constraints
    """
    b_0 = B[:,0]
    s = len(b_0)
    b_0.shape=(s,1)
    B_shifted = B-b_0

    N_dim = np.linalg.matrix_rank(B_shifted)
    if N_dim == s: #No need to transform using SVD, we can simply use qhull
        if N_dim == 1:
            N_hull,a_hull_sh = Eqsys_line(B_shifted)
        else:
            hull = ConvexHull(B_shifted.T)
            E = hull.equations
            N_hull=E[:,:-1].T
            a_hull_sh=E[:,-1]
            a_hull_sh.shape=[len(a_sh),1]

        N_subspace = np.array([[]])
        a_subspace_sh = np.array([[]])


    if N_dim <= s: #Transform using SVD
        print('Transform, n_dim =',N_dim)
        U,S,V = np.linalg.svd(B_shifted)
        B_transformed = U[:,0:N_dim].T@B_shifted
        if N_dim == 1:
            N_transformed,a_hull_sh = Eqsys_line(B_transformed)
        else:
            hull = ConvexHull(B_transformed.T)
            E = hull.equations
            N_transformed=E[:,:-1].T
            a_hull_sh=E[:,-1]
            a_hull_sh.shape=[len(a_hull_sh),1]


        N_hull=U[:,0:N_dim]@N_transformed

        N_subspace =U[:,N_dim:]
        a_subspace_sh = np.zeros((s-N_dim,1))

    a_hull = a_hull_sh-N_hull.T@b_0
    a_subspace = a_subspace_sh-N_subspace.T@b_0

    a_hull.shape = (len(a_hull),)
    a_subspace.shape = (len(a_subspace),)

    return (N_hull,a_hull,N_subspace,a_subspace)


def solve_LP_convex(solver,B,b_orig,rkm,u,K,dt,reduce = False,verbose_LP = False,minval = 0,maxval=np.infty, **options):
    """
    This method solves the LP Problem

    Parameters:

    solver:     a string with the solver that should be used
    O:          Order condition matrix
    rhs:        Order condition right hand side

    reduce:     If set, the LP problem is first solved with a reduced set of constriants
    verbose_LP: prints additional messages
    minval:     Vector or scalar, is the minimum to enforce for solution. If not needed set to -inf
    maxval:     Vector or scalar, is the minimum to enforce for solution. If not needed set to inf
    options:    additional Options, are passed through to the used solver

    Returns:
    status: integer representing the status of the algorithm.
      For scipy
        0 : Optimization proceeding nominally.
        1 : Iteration limit reached.
        2 : Problem appears to be infeasible.
        3 : Problem appears to be unbounded.
        4 : Numerical difficulties encountered.
      For cvypy
        0 : Optimization proceeding nominally.
        2 : Problem appears to be infeasible.
        4 : Numerical difficulties encountered.
        5 : solver crashed
        6 : Trivial Problem
    l = Array with number of constraints
    b: found b, if solver failed rkm.b

    """
    s = len(rkm.b)
    k = K.shape[0] #Number of ODEs

    l = []

    #Filter for variables with boundarys
    if np.isscalar(minval):
        minval = np.repeat(minval,k)
    if np.isscalar(maxval):
        maxval = np.repeat(maxval,k)

    i_n = minval > -np.infty
    i_p = maxval < np.infty


    i_min = np.zeros(k,bool)
    i_max = np.zeros(k,bool)
    if reduce: #We do not need it otherwise
        u_ = u +dt*K@rkm.b

    cnt_iter_h = 0
    while True: #We check at the end for break conditions
        cnt_iter_h += 1
        #Reduce
        if reduce:
            i_min[i_n] = i_min[i_n] | (u_[i_n] < minval[i_n])
            i_max[i_p] = i_max[i_p] | (u_[i_p] > maxval[i_p])

        else:
            i_min = i_n
            i_max = i_p

        if not np.any(i_min) and not np.any(i_max):
            print('trivial Problem')
            return (0,0,rkm.b)


        l.append(np.sum(i_min)+np.sum(i_max))

        #Solve LP-Problem

        u_min = u[i_min] #slice the u and K
        K_min = K[i_min,:]
        u_max = u[i_max]
        K_max = K[i_max,:]
        minval_min = minval[i_min]
        maxval_max = maxval[i_max]

        if solver == 'scipy_ip' or solver == 'scipy_sim':
            raise NotImplementedError

        else:
            #We wanrt to optimize for ||b-bt||_1=||

            ap_op =cp.Variable(s)
            an_op =cp.Variable(s)
            e = np.ones(s) #vector for goal Function, just generates the 1-Norm of b

            (N_hull,a_hull,N_subspace,a_subspace) = Eqsys_chull(B)

            if len(N_subspace) ==0:
                print('Do the B satisfy sum b = 1?, The b have to be at least of 1st order')
                raise NotImplementedError

            #Using different implementations to aviod empyt matricies
            if K_max.shape[0]>0 and K_min.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [N_subspace.T@(ap_op-an_op+b_orig)+a_subspace==0,
                    N_hull.T@(ap_op-an_op+b_orig)+a_hull<=0,
                    u_min+dt*K_min@(ap_op-an_op+b_orig)>=minval_min,
                    u_max+dt*K_max@(ap_op-an_op+b_orig)<=maxval_max,ap_op>=0,an_op>=0])
            elif K_min.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [N_subspace.T@(ap_op-an_op+b_orig)+a_subspace==0,
                    N_hull.T@(ap_op-an_op+b_orig)+a_hull<=0,
                    u_min+dt*K_min@(ap_op-an_op+b_orig)>=minval_min,ap_op>=0,an_op>=0])
            elif K_max.shape[0]>0:
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [N_subspace.T@(ap_op-an_op+b_orig)+a_subspace==0,
                    N_hull.T@(ap_op-an_op+b_orig)+a_hull<=0,
                    u_max+dt*K_max@(ap_op-an_op+b_orig)<=maxval_max,ap_op>=0,an_op>=0])
            else:
                raise ValueError
                #should alredy be detected as trivial Problem

            try:
                prob.solve(solver=solver,**options)
                b = ap_op.value - an_op.value + b_orig
            except:
                if verbose_LP: print('Solver crashed')
                return (5,l,None,0)
            if prob.status == cp.OPTIMAL:
                status = 0
            elif prob.status == cp.OPTIMAL_INACCURATE:
                if verbose_LP: print(prob.status)
                status = 4
            else:
                status = 5


        #Test for break conditions
        if np.all(i_min == i_n) and np.all(i_max == i_p): #reached whole set
            break
        u_ = u +dt*K@b
        if np.all(np.greater_equal(i_min,u_<minval)) and np.all(np.greater_equal(i_max,u_>maxval)):
            #if there are no new negative values appearing
            break

    return (status,l,b,cnt_iter_h)




def calculate_stages_imp(t,dt,u,rkm,f,solver_eqs,verbose=False,solveropts={}):
    """
    Function to calculate the stagevalues for a diagonaly implicit RKM

    Paramters:
    t:          time at beginning of step
    dt:         Timestep
    u:          solution
    rkm:        RKM to use
    f:          right hand side
    solver_eqs: solver to solve the equation system of the stageequations
    solveropts: optins for solver as dict
    verbose:    print additional messages to terminal

    Returns:
    K:          Matrix with the function evaluations of f. K = [f(t',u'),...,f(t',u')]
    message:    A status message as string
    neg_stage:  A vector with the length s where neg_stage[i] = 1 if negative values occured for u' at stage i
    """
    s= len(rkm)
    c= rkm.c
    A= rkm.A
    K = np.zeros([len(u),s])

    message = ''
    neg_stage = np.zeros(s)

    for i in range(s): #compute Stages
        u_prime = u.copy()
        for m in range(i):
            u_prime += dt*A[i,m]*K[:,m]

        K[:,i] = solver_eqs(t+c[i]*dt,u_prime,dt,A[i,i],f,**solveropts)

        if np.any(u_prime<-1.e-6):
            message = message + 'negative u\' at stage' + str(s) + '\n'
            neg_stage[i] = 1 #np.nonzero(u_prime > 1.e-6)
            if verbose: print('negative stagevalue found')
            if verbose >=2: print(i,u_prime) #print input to f(t,u) if it is negative
    return K,message,neg_stage



def calculate_stages_exp(t,dt,u,rkm,f,verbose=False):
    """
    Function to calculate the stagevalues for explicit RKM

    Paramters:
    t:          time at beginning of step
    dt:         Timestep
    u:          solution
    rkm:        RKM to use
    f:          right hand side
    verbose:    print additional messages to terminal

    Returns:
    K:          Matrix with the function evaluations of f. K = [f(t',u'),...,f(t',u')]
    message:    A status message as string
    neg_stage:  A vector with the length s where neg_stage[i] = 1 if negative values occured for u' at stage i
    """
    s= len(rkm)
    c= rkm.c
    A= rkm.A
    K = np.zeros([len(u),s])

    message = ''
    neg_stage = np.zeros(s)

    for i in range(s): #compute Stages
        u_prime = u.copy()
        for m in range(i):
            u_prime += dt*A[i,m]*K[:,m]

        K[:,i] = f(t+c[i]*dt,u_prime)

        if np.any(u_prime<-1.e-6):
            message = message+ 'negative u\' at stage' + str(s) + '\n'
            neg_stage[i] = 1 #np.nonzero(u_prime > 1.e-6)
            if verbose: print('negative stagevalue found')
            if verbose >=2: print(i,u_prime) #print input to f(t,u) if it is negative
    return K,message,neg_stage



def adapt_b(rkm,K,dt,u,minval,maxval,tol_neg,tol_change,p,theta,solver,solveropts,verbose = False):
    """
    function to adapt the b to meke sure it complies with the boundaries
    Parameters:
    rkm:        RKM used
    K:          Matrix with stagevalues
    dt:         dt used to calculate the stagevalues
    u:          solution at timestep
    minval:     Minimum value
    maxval:     Maximum value
    tol_neg:    Which negativevalues are accepted for u
    tol_change: MAximum value for |K@(b-rkm.b)|_2 accepted
    p:          range of orders to try enforcing as iterabel
    theta:      factors of timesteps to try as iterable. Element of [0 to 1]^k
    solver:     solver to use
    solveropts: optins for the LP-Problem
    verbose:    Print additional messages

    return:
    success:     True if a new b could be found
    u_n:          u^{n+1}
    b:          The b used
    dt:         the dt used
    message:    A status message as text
    status:     Status as dict

    """
    message = ''
    status = {'cnt_iter_h_max':0}
    change = None

    for i,the in enumerate(theta): #loop through all the sub-timesteps
        for p_new in p:     #loop through orders

            if verbose: print('Try: Order=',p_new,'Theta=',the)
            #Construct Order conditions
            O,rhs = OrderCond(rkm.A,rkm.c,order=p_new,theta=the)
            if the == 1:
                b_orig = rkm.b
            else:
                b_orig = rkm.b_dense(the)

            (status_LP,l,b,cnt_iter_h) = solve_LP(solver,O,rhs,rkm,b_orig,u,K,dt,maxval = maxval,minval = minval,**solveropts)
            status['cnt_iter_h_max'] = max(cnt_iter_h,status['cnt_iter_h_max'])

            if status_LP in [2,3,5]:
                #Error Handling for didn not work
                if verbose:    print('LP-Solve failed, probably infeasibel')
            else: #Did work, testing further
                u_n = u + dt*K@b
                if not (np.all(u_n >= minval-tol_neg) and np.all(u_n <= maxval+tol_neg)) :
                    #got a solution form the LP solver that is stil not positive...
                    #do some error handling here
                    if verbose:    print('LP-Solve returned a b that leads to a false solution')
                    if verbose >= 2:    print(min(u_n-minval)); print(max(u_n-maxval)); print(u_n)
                else:
                    change = dt*np.linalg.norm(K@(b-rkm.b))
                    if change > tol_change: # to big adaption...
                        #do some error handling here
                        if verbose:    print('a to big adaptation to the solution by changing the b')
                        if verbose >= 2: print('|K(b_new-b)|=',change)
                    else: #we got a acceptable solution
                        if verbose: print('found new b')
                        return True, u_n,b, dt*the, message, change,p_new,the,status

    return False, None,np.zeros_like(rkm.b)*np.nan, 0, message, change, 0, 0,status

def adapt_b_convex(rkm,K,dt,u,minval,maxval,tol_neg,tol_change,p,theta,solver,solveropts,verbose = False):
    """
    function to adapt the b to meke sure it complies with the boundaries
    Parameters:
    rkm:        RKM used
    K:          Matrix with stagevalues
    dt:         dt used to calculate the stagevalues
    u:          solution at timestep
    minval:     Minimum value
    maxval:     Maximum value
    tol_neg:    Which negativevalues are accepted for u
    tol_change: MAximum value for |K@(b-rkm.b)|_2 accepted
    p:          range of orders to try enforcing as iterabel
    theta:      factors of timesteps to try as iterable. Element of [0 to 1]^k
    solver:     solver to use
    solveropts: optins for the LP-Problem
    verbose:    Print additional messages

    return:
    success:     True if a new b could be found
    u_n:          u^{n+1}
    b:          The b used
    dt:         the dt used
    message:    A status message as text
    status:     Status as dict

    """
    message = ''
    status = {'cnt_iter_h_max':0}
    change = None
    print('changed')

    for i,the in enumerate(theta): #loop through all the sub-timesteps
        for p_new in p:     #loop through orders

            if verbose: print('Try: Order=',p_new,'Theta=',the)
            #Construct Problem
            #Matrix with the used methods
            B = [[rkm.b]]
            for order in range(rkm.p,p_new-1,-1):
                B.append(rkm.b_hat[order])


            B = np.concatenate(B).T
            if verbose >= 2: display(B)
            b_orig = rkm.b

            (status_LP,l,b,cnt_iter_h) = solve_LP_convex(solver,B,b_orig,rkm,u,K,dt,maxval = maxval,minval = minval,**solveropts)
            status['cnt_iter_h_max'] = max(cnt_iter_h,status['cnt_iter_h_max'])

            if status_LP in [2,3,5]:
                #Error Handling for didn not work
                if verbose:    print('LP-Solve failed, probably infeasibel')
            else: #Did work, testing further
                u_n = u + dt*K@b
                if not (np.all(u_n >= minval-tol_neg) and np.all(u_n <= maxval+tol_neg)) :
                    #got a solution form the LP solver that is stil not positive...
                    #do some error handling here
                    if verbose:    print('LP-Solve returned a b that leads to a false solution')
                    if verbose >= 2:    print(min(u_n-minval)); print(max(u_n-maxval)); print(u_n)
                else:
                    change = dt*np.linalg.norm(K@(b-rkm.b))
                    if change > tol_change: # to big adaption...
                        #do some error handling here
                        if verbose:    print('a to big adaptation to the solution by changing the b')
                        if verbose >= 2: print('|K(b_new-b)|=',change)
                    else: #we got a acceptable solution
                        if verbose: print('found new b')
                        return True, u_n,b, dt*the, message, change,p_new,the,status

    return False, None,np.zeros_like(rkm.b)*np.nan, 0, message, change, 0, 0,status



class Solver:
    def __init__(self, rkm = None,dt = None,t_final = None,b_fixed = None,tol_neg=None,
                tol_change=None,p=None,theta=None,solver=None, convex = False,
                LP_opts=None,solver_eqs = None,fail_on_requect = True):
        self.rkm = rkm #        Base Runge-Kutta method, in Nodepy format
        self.dt = dt#         time step size
        self.t_final = t_final #    final solution time
        self.b_fixed = b_fixed #    if True rkm.b are used
        self.tol_neg = tol_neg #    Which negativevalues are accepted for u
        self.tol_change = tol_change # Maximum value for |K@(b-rkm.b)|_2 accepted
        self.p = p#        range of orders to try enforcing as iterabel
        self.theta = theta#     factors of timesteps to try as iterable. Element of [0 to 1]^k
        self.solver = solver#    the solver used for solving the LP Problem
        self.convex = convex #Use a convex combiantion of given methods when adapting the b
        self.LP_opts = LP_opts#:    Dict containing options for LP-solver
        self.solver_eqs = solver_eqs
        self.fail_on_requect = fail_on_requect#if True breaks if there is no fesible b

    def __str__(self):
        string =    ("RKM:           " +self.rkm.name + "\n" +
                    "dt:            " +str(self.dt) + "\n" +
                    "t_final:       " +str(self.t_final) + "\n" +
                    "b_fixed:       " +str(self.b_fixed) + "\n" +
                    "tol_neg:       " +str(self.tol_neg) + "\n" +
                    "tol_change:    " +str(self.tol_change) + "\n" +
                    "p:             " +str(self.p) + "\n" +
                    "theta:         " +str(self.theta) + "\n" +
                    "solver:        " +str(self.solver) + "\n" +
                    "LP_opts:       " +str(self.LP_opts) + "\n" +
                    "solver_eqs:    " +str(self.solver_eqs) + "\n" +
                    "fail on re:    " +str(self.fail_on_requect) + "\n" )
        return string




class Problem:
    def __init__(self, f=None, u0 = None,minval = 0,maxval = np.infty,description = ''):
        self.f = f #        RHS of ODE system
        self.u0 = u0#         Initial data
        self.minval = minval#
        self.maxval = maxval#        Limits for Problem
        self.description = description

    def __str__(self):
        return self.description

class StepsizeControl:
    def __init__(self,dt_max,dt_min,a_tol,r_tol,f,tol_reqect):
        """
        Class to organize stepsize control

        Parmeters:
        dt_max:     maximum stepsie
        dt_min:     minimum stepsze
        f:          function of (dt,error) that returns the next stepsize

        """
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.f = f
        self.tol_reqect = tol_reqect



def RK_integrate(solver = [], problem = [],stepsize_control = None, dumpK=False,verbose=False):

    """
    Options:
        solver: Solver Object with the fields:
            rkm:        Base Runge-Kutta method, in Nodepy format
            dt:         time step size
            t_final:    final solution time
            b_fixed:    if True rkm.b are used
            tol_neg:    Which negativevalues are accepted for u
            tol_change: Maximum value for |K@(b-rkm.b)|_2 accepted
            p:          range of orders to try enforcing as iterabel
            theta:      factors of timesteps to try as iterable. Element of [0 to 1]^k
            solver:     the solver used for solving the LP Problem
            LP-opts:    Dict containing options for LP-solver
            solver_eqs: Solver for stageeqation for implicit method


        problem: Problem object with the fields:
            f:          RHS of ODE system
            u0:         Initial data
            minval:
            maxval:        Limits for Problem



        stepsize_control: (optional) StepsizeControl object with the fields:
            Parmeters:
            dt_max:     maximum stepsie
            dt_min:     minimum stepsze
            tol:        tolerance
            f:          function of (stepsize_control,dt_old,error,change,success) that returns the next dt



        dumpK:      if True the stage values are also returned
        verbose:    if True function prints additional messages


    Returns:
        u:      Matrix with the solution
        t:      vector with the times
        b:      Matrix with the used b's

        if dumpK = True:
         K:     Array of the K Matrix containing the stagevalues
        if return_status
         status: dict containing
                'dt': the used dt
                'success': True if solver succeded, False if a inveasible or illdefined LP-Problem occured or the solver crashed
                'message': String containing more details
                'b':       Array with the indecies where b was changed
    """

    if not 'verbose_LP' in solver.LP_opts.keys():
        solver.LP_opts['verbose_LP'] = verbose


    #setup Variables for Soulution storage
    uu = [problem.u0]
    tt = [0]


    #setup Problem Solve
    explicit = solver.rkm.is_explicit()
    t = 0
    u = problem.u0
    dt= solver.dt

    #setup stepsize control
    dt_old = dt #variable with the last tried dts
    dt_adp = dt
    error = None
    tol_met = True

    success = True #For stepsize control at first step


    if dumpK:
        KK = ['null']


    #for debbugging bs
    bb = [solver.rkm.b]

    status = {
        'Solver':  solver,
        'Problem': problem,
        'success': True,
        'message': '',
        'neg_stage': [None],
        'LP_stat': [None],
        'b':[None],
        'sc':[None],
        'change':[None],
        'old_min':[None],
        'new_min':[None],
        'old_max':[None],
        'new_max':[None],
        'order':[None],
        'theta':[None],
        'dt_calc':[None],
        'error':[None]
    }
    if verbose: print('set up, starting to solve')

    while t<solver.t_final:
        #Control new stepsize
        if t>0 and stepsize_control: #not call on startup
            dt = stepsize_control.f(stepsize_control,dt_old,dt_adp,error,change,success,tol_met) #dt is the dt with adaption
            dt_old = dt
        else:
            dt = solver.dt
        if t+dt > solver.t_final:
            dt = solver.t_final-t




        #Compute the new K
        if verbose: print('calculation new set of stagevalues for t =',t,'dt=',dt)
        if explicit:
            K,message,neg_stage = calculate_stages_exp(t,dt,u,solver.rkm,problem.f,verbose=verbose)
        else:
            K,message,neg_stage = calculate_stages_imp(t,dt,u,solver.rkm,problem.f,solver.solver_eqs,verbose=verbose)
        status['message'] += message
        status['neg_stage'].append(neg_stage)
        status['dt_calc'].append(dt)



        if dumpK:
            KK.append(K)

        #compute initial guess
        u_n = u +dt*K@solver.rkm.b

        if solver.b_fixed:
            u_n = u_n
            b = solver.rkm.b
            status['b'].append('o')
            change = 0
        else:
            if np.all(u_n >= problem.minval) and np.all(u_n <= problem.maxval):
                #everything is fine
                b = solver.rkm.b
                success = True
                status['b'].append('o')
                status['LP_stat'].append(None)
                status['change'].append(None)
                status['order'].append(None)
                status['theta'].append(None)
                status['old_min'].append(None)
                status['new_min'].append(None)
                status['old_max'].append(None)
                status['new_max'].append(None)
                change = 0
            else:
                status['old_min'].append(np.min(u_n - problem.minval))
                status['old_max'].append(np.max(u_n - problem.maxval))
                if solver.convex:
                    success,u_n,b,dt,message, change, order, the,LP_stat = adapt_b_convex(solver.rkm,K,dt,u,problem.minval,problem.maxval,
                        solver.tol_neg,solver.tol_change,solver.p,solver.theta,solver.solver,solver.LP_opts,verbose = verbose)
                else:
                    success,u_n,b,dt,message, change, order, the,LP_stat = adapt_b(solver.rkm,K,dt,u,problem.minval,problem.maxval,
                        solver.tol_neg,solver.tol_change,solver.p,solver.theta,solver.solver,solver.LP_opts,verbose = verbose)
                status['b'].append('c')
                status['message'] += message
                status['LP_stat'].append(LP_stat)
                status['change'].append(change)
                status['order'].append(order)
                status['theta'].append(the)
                if success:
                    status['new_min'].append(np.min(u_n - problem.minval))
                    status['new_max'].append(np.max(u_n - problem.maxval))
                else:
                    status['new_min'].append(None)
                    status['new_max'].append(None)

        #Calculate error estimate
        if stepsize_control:
            #if success:
            #    sc = stepsize_control.a_tol+np.max([u,u_n],0)*stepsize_control.r_tol
            #else:
            #    sc = stepsize_control.a_tol+u*stepsize_control.r_tol
            #error = np.linalg.norm((K@(solver.rkm.b-solver.rkm.bhat))/sc)/len(u)
            error = dt_old*np.linalg.norm((K@(solver.rkm.b-solver.rkm.bhat)))
            status['error'].append(error)
            if verbose: print('Error:',error)
            if error > stepsize_control.tol_reqect:
                tol_met = False
                if verbose: print('tol not met')
                status['message'] += 'tol not met'
                if dt_old <= stepsize_control.dt_min+1e-15:
                    if verbose: print('dt_min reached, using it anyway')
                    status['sc'].append('m')
                else:
                    status['sc'].append('r')
            else:
                tol_met = True
                status['sc'].append('m')


        if success and (tol_met or dt_old <= stepsize_control.dt_min+1e-15):
            if verbose: print('advancing t')
            t += dt
            u = u_n
            ##Cropp negative Values TODO more options/Dokumentation
            #u[u<0] = 0

        if not success:
            if verbose: print('step reqect')
            if solver.fail_on_requect:
                status['success'] = False
                status['b'][-1] = 'r'
                break
            else:
                status['b'][-1] = 'r'



        bb.append(b)
        uu.append(u)
        tt.append(t)


    ret = (status,tt,uu,bb)
    if dumpK:
        ret= ret + (KK,)


    return ret











#For Implicit Methods

#Define a solver for the equation system
def solver_Matrix(t,u,dt,a,A,preconditioner = None,verbose_solver = False):
    """
    The function solves a equation system of the Form
    x = f(t,u+dt*a*x)
    and returns x
    where f(t,u)=Au
    """
    x = np.linalg.solve(dt*a*A-np.eye(u.size),-A@u)
    #print(max(abs((dt*a*A-np.eye(u.size))@x+A@u)))
    return x






def solver_nonlinear(t,u,dt,a,f,verbose_solver = False):
    """
    The function solves a equation system of the Form
    x = f(t,u+dt*a*x)
    and returns x

    f is a function of t and u
    """
    stageeq = lambda x: f(t,u+dt*a*x)-x  # it seems like solving for the argument is better
    x, info, ier, mesg = opt.fsolve(stageeq,u,full_output=1)
    if ier != 1:
        print(mesg)
    return x

def solver_nonlinear_arg(t,u,dt,a,f,verbose_solver = False,preconditioner=None):
    """
    The function solves a equation system of the Form
    x = f(t,u+dt*a*x)
    and returns x

    f is a function of t and u

    preconditioner: method for getting a starting point for fsolve (in terms of an y=u_start)
    """
    if preconditioner != None:
        y_0 = preconditioner(t,u,dt,a,f)
    else:
        y_0 = u

    #print('res orig:',np.linalg.norm(-u+u+dt*a*f(t,u)))
    #print('res new:',np.linalg.norm(-y_0+u+dt*a*f(t,y_0)))

    stageeq = lambda y: -y+u+dt*a*f(t,y)
    y, info, ier, mesg = opt.fsolve(stageeq,y_0,full_output=1)

    #check if solution is exact
    if np.any(np.abs(-y+u+dt*a*f(t,y))>0.0001):
        print('stageeq. solved non accurate')
        print(np.linalg.norm(-y+u+dt*a*f(t,y)))

    if np.any(u+dt*a*f(t,y)<0) and verbose_solver or np.any(u+dt*a*f(t,y)<-1e-8) :
        print('stageq solved with negative argument')
        print('res:')
        print(max(np.abs(-y+u+dt*a*f(t,y))))
        #print(u+dt*a*f(t,y))
        print('min:')
        print(min(u+dt*a*f(t,y)))

    ##Added here some stuff to test behavior of LP-Solver not ment for production!!!
    #Crops the negative values
    #h = f(t,y)
    #h[y<0]= u[y<0]/(dt*a)#We define y to 0 and calculate h using the stageeq

    #if ier != 1:
    #    print(mesg)
    #return(h)

    if ier != 1:
        print(mesg)
    return(f(t,y))

def solver_nonlinear_nk(t,u,dt,a,f,verbose_solver = False,preconditioner=None):
    """
    The function solves a equation system of the Form
    x = f(t,u+dt*a*x)
    and returns x

    f is a function of t and u

    The method uses the Newton-Krylov solver from scipy
    """
    if preconditioner != None:
        y_0 = preconditioner(t,u,dt,a,f)
    else:
        y_0 = u

    #print('res orig:',np.linalg.norm(-u+u+dt*a*f(t,u)))
    #print('res new:',np.linalg.norm(-y_0+u+dt*a*f(t,y_0)))

    stageeq = lambda y: -y+u+dt*a*f(t,y)

    y = opt.newton_krylov(stageeq,y_0)

    #check if solution is exact
    if np.any(np.abs(-y+u+dt*a*f(t,y))>1e-10):
        print('stageeq. solved non accurate')
        print(np.linalg.norm(-y+u+dt*a*f(t,y)))

    if np.any(u+dt*a*f(t,y)<0) and verbose_solver or np.any(u+dt*a*f(t,y)<-1e-8) :
        print('stageq solved with negative argument')
        print('res:')
        print(max(np.abs(-y+u+dt*a*f(t,y))))
        #print(u+dt*a*f(t,y))
        print('min:')
        print(min(u+dt*a*f(t,y)))

    return(f(t,y))
