"""
This file contains utility functions for investigating properties of the rk methods

plot_cnvergence: function to plot the convergence of methods

find_dt: method for investigating certain dts using a bisect algorithm

"""

import numpy as np
import matplotlib.pyplot as plt
from RKimple import RK_integrate


def plot_convergence(problem,solver,dt,refernce,step=1,error='abs',dx='1',Norm = 2,Params = {},get_order=False):
    """"
    Parameters:
    problem:            Problem Objects
    solver:             Solver object
    dt:                 dt array with dts
    reference:          Array with reference solutions to compare the computet solution against
    error:              Definition of error computation, one of 'abs','rel','grid'
    dx:                 Discretisation for grid norm
    Norm:               Norm to use for Error calculation ||u-u'||_Norm
    Params:             Parameters for time integrator
    get_order:          If True: also returns list with lowest orders

    Return:
    sol:                Array with the solutions used for calculationg the errors
    err:                Array with errors
    change:             Array, True if b was changed False otherwise

    """

    err = np.zeros_like(dt)
    change = np.zeros_like(dt,dtype=bool)
    order = np.zeros_like(dt)
    sol = []


    for i in range(dt.size):
        print('dt='+str(dt[i]))
        solver.dt = dt[i]
        status,t,u,b = RK_integrate(solver=solver,problem=problem,**Params)
        dif = refernce[:,i]-u[step]
        change[i] = 'c' in status['b']
        if status['success']:
            if error == 'abs':
                err[i] = np.linalg.norm(dif,ord=Norm)
            elif error == 'rel':
                err[i] = np.linalg.norm(dif,ord=Norm)/np.linalg.norm(refernce[:,i],ord=Norm)
            elif error == 'grid': #Grid function Norm (LeVeque Appendix A.5)
                err[i] = dx**(1/Norm)*np.linalg.norm(dif,ord=Norm)
            else:
                print('Error not defined')
                print(error)
                raise ValueError

            orders = np.array(status['order'])[np.array(status['order'])!=None]
            if len(orders)>0:
                order[i]=np.min(orders)
            else:
                order[i] = np.nan
        else:
            err[i] = np.nan
            order[i] = np.nan
        sol.append(u[step])

    plt.plot(dt,err,'o-')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.ylabel('Error')
    plt.xlabel('dt')

    if get_order:
        return sol,err,change,order
    else:
        return sol,err,change



def find_dt(problem,solver,dt_start,cond = '',tol = 0.001,Params = {}):

    """"
    The function searchs for the upper bound of dt that sattisfies a condition.
    A bisect search approach is used

    Parameters:
    problem:            Problem Objects
    solver:             Solver object
    dt_start:           dt to start with

    cond: which condition to search for one of:
            'dt_pos'
            'dt_stable'
            'dt_feasible'

    tol:                the toleranc for dt
    Params:             Parameters for time integrator

    Returns:
    dt_sol:             The wanted dt
    dt:                 Array with the tried timesteps
    val:                Array with information if the condition was fullfiled for the timesteps

    """
    #Check the settings for the integator


    if cond in ['dt_pos','dt_stable']:
        solver.b_fixed = True
    elif cond in ['dt_feasible']:
        solver.b_fixed = False
    else:
        print('no knwn conition')
        raise ValueError


    dt = np.array([0])
    val = np.array([True])

    run = 0 # number of iteration

    while True:
        #calculate new timestep
        if run == 0:
            dt_new = dt_start
        elif len(dt[~val]) == 0: #No failed run so far
            dt_new = 2*dt[-1]
        else:
            if val[-1] == False: #Last try failed
                inter = (max(dt[val]), dt[-1]) #between the highest succesfull run and the last run
            else: #last try succeded
                inter = (dt[-1], min(dt[~val])) #between last run and lowest failed run

            dt_new = 0.5 * np.sum(inter)

        dt = np.append(dt,dt_new)

        #run integration
        print('Testing:',dt[-1])
        solver.dt = dt[-1]
        #t,u,b,status = time_integrator(rkm,dt[-1],f,u0,dumpK=False,return_status = True,**Params)
        status,t,u,b = RK_integrate(solver=solver,problem=problem,dumpK=False,**Params)

        #Test if condition is met
        if cond == 'dt_pos':
            if np.all(np.array(u) > -1e-8):
                val_new = True
            else:
                val_new = False

        elif cond == 'dt_feasible':
            if status['success']:
                val_new = True
            else:
                val_new = False

        elif cond == 'dt_stable':
            n_start = np.linalg.norm(u[0])
            n_end = np.linalg.norm(u[-1])

            if n_end < 100*n_start:
                val_new = True
            else:
                val_new = False
        else:
            print('no knwn conition')
            raise ValueError

        val = np.append(val,val_new)
        print(val_new)

        #Test if we alredy know the solution
        if len(dt[~val]) > 0:
            if abs(min(dt[~val]) - max(dt[val])) < tol:
                dt_sol = max(dt[val])
                break

        run += 1
        if run == 500:
            print('500 iterations reached')
            raise ValueError

        if dt_new >= dt_start*1e5:
            print('Time is getting to big. Apparently the time is not valid. Setting to 0')
            dt_sol = 0
            break

    return (dt_sol,dt,val)


def findall_dt(problem,solver,dt_start,tol = 0.001,Params = {}):
    """"
    This function seachs for all important dt for a RKM.
    These are 'dt_pos','dt_stable' and 'dt_feasible'
    The find_dt() method is used


    Parameters:
    problem:            Problem Objects
    solver:             Solver object, if solver is tuple of Solver object search for all
    dt_start:           dt to start with
    tol:                the toleranc for dt

    Params:             Parameters for time integrator

    Returns:
    dt:                 dict with the dt's
    """

    conds = ('dt_pos','dt_stable','dt_feasible')
    dt = {}
    if not type(solver) is tuple:
        for cond in conds:
            print('search for',cond)
            dt_sol,dt_,val_ = find_dt(problem,solver,dt_start,cond = cond,tol = tol,Params = Params)
            dt[cond] = dt_sol
        return dt

    else: #check for more methods
        times = ()
        for rkm_ in solver:
            print(rkm_)
            dt = findall_dt(problem,rkm_,dt_start,tol = tol,Params = Params)
            print(dt)
            times = times + (dt,)
        return times



def plot_times(methods,dt,effective = False,title = ''):
    """"
    Function to plot the dt for multiple methods.

    Paramters:
    methods:    tuple with the methods
    dt:         tuple with dicts of the methods
    effective:  If true plot the effective timesteps
    title:      String as title for the plot


    """


    labels = []
    stages = []
    dt_pos = []
    dt_stable = []
    dt_feasible = []

    for i in range(len(methods)):
        rkm = methods[i]
        labels.append(rkm.name)
        stages.append(len(rkm))
        dt_pos.append(dt[i]['dt_pos'])
        dt_stable.append(dt[i]['dt_stable'])
        dt_feasible.append(dt[i]['dt_feasible'])


    stages = np.array(stages)
    dt_pos = np.array(dt_pos)
    dt_stable = np.array(dt_stable)
    dt_feasible = np.array(dt_feasible)


    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    if effective:
        rects1 = ax.bar(x - width, dt_pos/stages, width, label='dt_pos_eff')
        rects2 = ax.bar(x , dt_stable/stages, width, label='dt_stable_eff')
        rects3 = ax.bar(x + width, dt_feasible/stages, width, label='dt_feasible_eff')
    else:
        rects1 = ax.bar(x - width, dt_pos, width, label='dt_pos')
        rects2 = ax.bar(x , dt_stable, width, label='dt_stable')
        rects3 = ax.bar(x + width, dt_feasible, width, label='dt_feasible')

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid()
    print(labels)


def get_max_iter_h(status):
    max_iter = 0
    for lp_stat in status['LP_stat']:
        if lp_stat:
            max_iter = max(max_iter,lp_stat['cnt_iter_h_max'])
    return max_iter

def show_status(status):
    #print success
    print('Succesfull:'); print(status['success'])
    #number of adaptations
    print('number of adaptations:');print(status['b'].count('c'))
    #Number of step rejects
    print('Number of step rejects:');print(status['b'].count('r'))

    fig, axs = plt.subplots(5,1)

    #plot on which timesteps the b was adabpted

    axs[0].eventplot(np.nonzero(np.array(status['b'])=='c'), colors='blue', lineoffsets=0.5,
                    linelengths=0.5)

    #plot the step rejects
    axs[0].eventplot(np.nonzero(np.array(status['b'])=='r'), colors='red', lineoffsets=-0.5,
                    linelengths=0.5)

    #plot step reject caused by tol
    axs[0].eventplot(np.nonzero(np.array(status['sc'])=='r'), colors='red', lineoffsets=-1.5,
                    linelengths=0.5)

    axs[0].set_xlim([0,len(status['b'])])

    #plot the resulting order

    axs[1].plot([-1,len(status['b'])+1],[status['Solver'].rkm.order(),status['Solver'].rkm.order()],color = 'black')
    axs[1].plot(np.array(status['order']),'x',color = 'blue')
    axs[1].set_xlim([0,len(status['order'])])

    axs[1].set_ylim([-0.5,status['Solver'].rkm.order()+0.5])

    #plot the theta

    axs[2].plot([-1,len(status['b'])+1],[1,1],color = 'black')
    axs[2].plot(np.array(status['theta']),'x',color = 'blue')
    axs[2].set_xlim([0,len(status['order'])])

    axs[2].set_ylim([-0.1,1.1])

    #plot negative violations
    axs[3].plot(np.array(status['old_min']),'x')
    axs[3].plot(np.array(status['new_min']),'x')
    axs[3].set_xlim([0,len(status['order'])])

    #plot posifive violations
    axs[4].plot(np.array(status['old_max']),'x')
    axs[4].plot(np.array(status['new_max']),'x')
    axs[4].set_xlim([0,len(status['order'])])
