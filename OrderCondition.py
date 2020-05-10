#This file defines Order conditions for Runge Kutta methods in the form O@b = r


import numpy as np

def tau(k,A,c): 
    #generates tau vector 
    return 1/ np.math.factorial(k)*c**k - 1/np.math.factorial(k-1) *A @ c**(k-1)
    
def tau_hat(k,A,c):
    return c**k-k*A@(k-1)


def OrderCond(A,c,order = 1,theta = 1):
    """
    #Generates Order Condition Matrix O and right side vector r for Linear Equation System O@b=r
    
    example:
    
    A: A matrix of RKM
    c: c vector of RKM
    order: Order to compute the order condition
    
    (O,rhs) = OrderCond(rkm.A,rkm.c,order = 1)
    
    O@b - rhs #should return 0
    
    """
    
    s = len(c) #number of stages
    
    r = []
    O_rows = []
    
    
    if A.shape != (s,s):
        raise ValueError
        
    else:
        if order >= 1:
            O_rows.append(np.ones(s));      r.append(theta)
            
        if order >= 2:
            O_rows.append(c);               r.append(theta**2/2)
            
        if order >= 3:
            O_rows.append(c**2);            r.append(theta**3/3)
            O_rows.append(tau(2,A,c));      r.append(0.)
            
        if order >= 4:
            O_rows.append(c**3);            r.append(theta**4/4)
            O_rows.append(tau(2,A,c)*c);    r.append(0.)
            O_rows.append(tau(2,A,c)@A.T);  r.append(0.)
            O_rows.append(tau(3,A,c));      r.append(0.)
        
        if order >= 5:
            O_rows.append(c**4);                     r.append(theta**5/5)
            O_rows.append(A@np.diag(c)@tau(2,A,c));  r.append(0.)
            O_rows.append(A@A@tau(2,A,c));           r.append(0.)
            O_rows.append(A@tau(3,A,c));             r.append(0.)
            O_rows.append(tau(4,A,c));               r.append(0.)
            O_rows.append(np.diag(c)@A@tau(2,A,c));  r.append(0.)
            O_rows.append(np.diag(c)@tau(3,A,c));    r.append(0.)
            O_rows.append(np.diag(c**2)@tau(2,A,c)); r.append(0.)
            O_rows.append(tau(2,A,c)**2);            r.append(0.)
        if order >= 6:
            if theta != 1: 
                print('no dense output for that order')
                raise NotImplementedError
            # order 6 conditions:
            O_rows.append(np.dot(A,c**4));                             r.append(1/30.)
            O_rows.append(np.dot(A,np.dot(A,c**3)));                   r.append(1/120.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c**2))));         r.append(1/360.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,np.dot(A,c)))));  r.append(1/720.)
            O_rows.append(np.dot(A,np.dot(A,c*np.dot(A,c))));          r.append(1/240.)
            O_rows.append(np.dot(A,c*np.dot(A,c**2)));                 r.append(1/90.)
            O_rows.append(np.dot(A,c*np.dot(A,np.dot(A,c))));          r.append(1/180.)
            O_rows.append(np.dot(A,c**2*np.dot(A,c)));                 r.append(1/60.)
            O_rows.append(np.dot(A,np.dot(A,c)*np.dot(A,c)));          r.append(1/120.)
            O_rows.append(np.dot(A,c**3)*c);                           r.append(1/24.)
            O_rows.append(np.dot(A,np.dot(A,c**2))*c);                 r.append(1/72.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c)))*c);          r.append(1/144.)
            O_rows.append(np.dot(A,c*np.dot(A,c))*c);                  r.append(1/48.)
            O_rows.append(np.dot(A,c**2)*c*c);                         r.append(1/18.)
            O_rows.append(np.dot(A,np.dot(A,c))*c*c);                  r.append(1/36.)
            O_rows.append(np.dot(A,c)*c**3);                           r.append(1/12.)
            O_rows.append(np.dot(A,c**2)*np.dot(A,c));                 r.append(1/36.)
            O_rows.append(np.dot(A,np.dot(A,c))*np.dot(A,c));          r.append(1/72.)
            O_rows.append(np.dot(A,c)*np.dot(A,c)*c);                  r.append(1/24.)
            O_rows.append(c**5);                                       r.append(1/6.)
        if order >= 7:
            if theta != 1: 
                print('no dense output for that order')
                raise NotImplementedError
            # order 7 conditions:
            O_rows.append(np.dot(A,c**5));                                          r.append(1/42.)
            O_rows.append(np.dot(A,np.dot(A,c**4)));                                r.append(1/210.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c**3))));                      r.append(1/840.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,np.dot(A,c**2)))));            r.append(1/2520.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,np.dot(A,np.dot(A,c))))));     r.append(1/5040.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c*np.dot(A,c)))));             r.append(1/1680.)
            O_rows.append(np.dot(A,np.dot(A,c*np.dot(A,c**2))));                    r.append(1/630.)
            O_rows.append(np.dot(A,np.dot(A,c*np.dot(A,np.dot(A,c)))));             r.append(1/1260.)
            O_rows.append(np.dot(A,np.dot(A,c**2*np.dot(A,c))));                    r.append(1/420.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c)*np.dot(A,c))));             r.append(1/840.)
            O_rows.append(np.dot(A,c*np.dot(A,c**3)));                              r.append(1/168.)
            O_rows.append(np.dot(A,c*np.dot(A,np.dot(A,c**2))));                    r.append(1/504.)
            O_rows.append(np.dot(A,c*np.dot(A,np.dot(A,np.dot(A,c)))));             r.append(1/1008.)
            O_rows.append(np.dot(A,c*np.dot(A,c*np.dot(A,c))));                     r.append(1/336.)
            O_rows.append(np.dot(A,c**2*np.dot(A,c**2)));                           r.append(1/126.)
            O_rows.append(np.dot(A,c**2*np.dot(A,np.dot(A,c))));                    r.append(1/252.)
            O_rows.append(np.dot(A,c**3*np.dot(A,c)));                              r.append(1/84.)
            O_rows.append(np.dot(A,np.dot(A,c)*np.dot(A,c**2)));                    r.append(1/252.)
            O_rows.append(np.dot(A,np.dot(A,c)*np.dot(A,np.dot(A,c))));             r.append(1/504.)
            O_rows.append(np.dot(A,c*np.dot(A,c)*np.dot(A,c)));                     r.append(1/168.)
            O_rows.append(np.dot(A,c**4)*c);                                        r.append(1/35.)
            O_rows.append(np.dot(A,np.dot(A,c**3))*c);                              r.append(1/140.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c**2)))*c);                    r.append(1/420.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,np.dot(A,c))))*c);             r.append(1/840.)
            O_rows.append(np.dot(A,np.dot(A,c*np.dot(A,c)))*c);                     r.append(1/280.)
            O_rows.append(np.dot(A,c*np.dot(A,c**2))*c);                            r.append(1/105.)
            O_rows.append(np.dot(A,c*np.dot(A,np.dot(A,c)))*c);                     r.append(1/210.)
            O_rows.append(np.dot(A,c**2*np.dot(A,c))*c);                            r.append(1/70.)
            O_rows.append(np.dot(A,np.dot(A,c)*np.dot(A,c))*c);                     r.append(1/140.)
            O_rows.append(np.dot(A,c**3)*c*c);                                      r.append(1/28.)
            O_rows.append(np.dot(A,np.dot(A,c**2))*c*c);                            r.append(1/84.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c)))*c*c);                     r.append(1/168.)
            O_rows.append(np.dot(A,c*np.dot(A,c))*c*c);                             r.append(1/56.)
            O_rows.append(np.dot(A,c**2)*c**3);                                     r.append(1/21.)
            O_rows.append(np.dot(A,np.dot(A,c))*c**3);                              r.append(1/42.)
            O_rows.append(np.dot(A,c)*c**4);                                        r.append(1/14.)
            O_rows.append(np.dot(A,c**2)*np.dot(A,c**2));                           r.append(1/63.)
            O_rows.append(np.dot(A,np.dot(A,c))*np.dot(A,c**2));                    r.append(1/126.)
            O_rows.append(np.dot(A,np.dot(A,c))*np.dot(A,np.dot(A,c)));             r.append(1/252.)
            O_rows.append(np.dot(A,c**3)*np.dot(A,c));                              r.append(1/56.)
            O_rows.append(np.dot(A,np.dot(A,c**2))*np.dot(A,c));                    r.append(1/168.)
            O_rows.append(np.dot(A,np.dot(A,np.dot(A,c)))*np.dot(A,c));             r.append(1/336.)
            O_rows.append(np.dot(A,c*np.dot(A,c))*np.dot(A,c));                     r.append(1/112.)
            O_rows.append(np.dot(A,c**2)*np.dot(A,c)*c);                            r.append(1/42.)
            O_rows.append(np.dot(A,np.dot(A,c))*np.dot(A,c)*c);                     r.append(1/84.)
            O_rows.append(np.dot(A,c)*np.dot(A,c)*c*c);                             r.append(1/28.)
            O_rows.append(np.dot(A,c)*np.dot(A,c)*np.dot(A,c));                     r.append(1/56.)
            O_rows.append(c**6);                                                    r.append(1/7.)
        if order >=8:
            print('too high order')
            raise NotImplementedError
        


        O = np.vstack(O_rows)
        return (O,np.array(r))
            
                

