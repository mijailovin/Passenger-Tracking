import numpy as np
import matplotlib.pyplot as plt


def trajectories1():
    """Generates the trajectories for exercise 1.

    Returns
    -------
    dict
        A dict with the trajectories.

    """
    T=1 # sample time
    # Trajectory 1
    N1=round(2000/(100/3.6*T))
    X1=np.zeros((2, N1), dtype=float)
    X1[1,:]=1000
    X1[0,:]= np.linspace(0, 2000, N1)
    
    # Trajectory 2
    r=200
    N2=round(r*np.pi/(50/3.6)) #recimo, treba prema brzini
    alpha=np.linspace(0, np.pi, N2)
    X2=np.zeros((2, N2), dtype=float)
    X2[0, :]=2000+r*np.sin(alpha)
    X2[1, :]=800+r*np.cos(alpha)
        
    # Trajectory 3
    N3=round(2000/(70/3.6*T)) #brzina 70kmph
    X3=np.zeros((2, N3), dtype=float)
    X3[1,:]=600
    X3[0,:]= np.linspace(2000, 0, N3)
    
    # Trajectory 4
    X4 = np.concatenate([X1, X2[:, 1:], X3[:, 1:]], axis=1)
    return dict(X1=X1, X2=X2, X3=X3, X4=X4)

def trajectories2():
    """Generates the trajectories for exercise 2.

    Returns
    -------
    dict
        A dict with the trajectories.

    """
    T = trajectories1()
    
    N5 = int(2000/(100/3.6))
    X5 = np.zeros((2, N5), dtype=float)
    p0 = np.array([200, 1200])
    pT = p0+2000*np.array([np.cos(-np.pi/6), np.sin(-np.pi/6)])
    X5[0, :] = np.linspace(p0[0], pT[0], N5)
    X5[1, :] = np.linspace(p0[1], pT[1], N5)


    N6 = int(2000/(100/3.6))
    X6 = np.zeros((2, N6), dtype=float)
    X6[0, :] = np.linspace(1500, 1500, N6)
    X6[1, :] = np.linspace(1800, 800, N6)
    
    T['X5']=X5
    T['X6']=X6
    return T

def add_noise(trajectories, Pd, clutter_model, rng=None):
    #prazna lista ako nista nije izmereno u tom merenju
    mean=np.array([0, 0])
    cov=np.array([[.1, 0], 
                  [0, .1]])
    Y=[]
    N=max([T.shape[1] for key, T in trajectories.items()]) 
    # N = najduza sekvenca od svih putanja
    if rng is None:
        rng = np.random.default_rng()
    
    for n in range(N):
        y= np.array([])
        
        trajs = [T for key, T in trajectories.items() if n<T.shape[1]]
        # samo trajektorije koje su aktivne u trenutku n
        for i, traj in enumerate(trajs):
            r= rng.random()
            if r<Pd: #generise se neko tacno merenje
                new = traj[:, n:n+1] + rng.multivariate_normal(mean, cov, 1).reshape((2,1))
                if y.any():
                    y = np.concatenate((y, new), axis=1)
                else:
                    y = new
            
        #num = koliko cluttera ima za trenutna merenja
        num= rng.poisson(lam = clutter_model['lam']) #lam = beta*V
        # num=0
        # print(len(trajs), num)
        xmin=clutter_model['volume']['xmin'];  xmax=clutter_model['volume']['xmax'];
        ymin=clutter_model['volume']['ymin'];  ymax=clutter_model['volume']['ymax'];
        for j in range(num): 
            new = np.array([[rng.uniform(xmin, xmax)], \
                            [rng.uniform(ymin, ymax)]]  )
            if y.any():
                y = np.concatenate((y, new), axis=1)
            else:
                y = new
            
        #print(y)
        Y.append(y)
        
    return Y

def path_plot(T, Y, X_est=None):
    plt.figure()
    for key, X in T.items():
        # print(key, item.shape)
        p1, =plt.plot(X[0,:], X[1,:], 'r', linewidth=1)
        for i in range(len(Y)): 
            if Y[i].any():            
                p2 = plt.scatter(Y[i][0, :], Y[i][1,:], marker='o', color='b', s=0.5)

            
                
    
    if not X_est is None:
        p3 =plt.scatter(X_est[0,:], X_est[1, :], marker='x', color='g')
        plt.xlim([-50, 2500])
        plt.ylim([550,1050])
    
        plt.legend(handles = [p1, p2, p3], 
                labels  = ['Tačna putanja', 'Merenja', 'Estimacija putanje'])
    else:
        plt.legend(handles = [p1, p2], 
                  labels  = ['Tačna putanja', 'Merenja'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
#%%

T = trajectories1()
PD = 0.9 # sensor
lam = 2
volume = dict(xmin=-100, xmax=2500, ymin=0, ymax=1100)
clutter_model = dict(volume=volume, lam=lam)

Y=add_noise({'T': T['X1']}, PD, clutter_model, rng=np.random.default_rng(13))

path_plot({'T': T['X1']}, Y)

# =============================================================================
# ostale bitne funkcije
# =============================================================================


#%% Kalman filter for single target tracking

def cv_model(Q, T):
    """A constant velocity model.

    Parameters
    ----------
    Q : numpy.ndarray
        The process noise covariance.
    T : float
        Sampling interval

    Returns
    -------
    dict
        A dict with the motion model.

    """
    # D - dimensions =2
    # T - sampling time
    # CV model
    F=np.array([[1, 0, T, 0],
                [0, 1, 0, T],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    G=np.array([[1, 0],
                [0, 0.5],
                [1, 0],
                [0, 1]])
    f = lambda x: F@x
    motion_model = dict(f=f, F=F, Q=G @ Q @ G.T)
    
    return motion_model


    
class KF:
    def __init__(self, motion_model, R):
        """An implementation of an  Kalman Filter.

        Parameters
        ----------
        motion_model : dict
            The motion model to use for the filtering.
        R: np.ndarray
           kovarijaciona matrica za senzor
        """
        self.motion_model = motion_model
        self.R=R 

    def propagate(self, x, P):
        """Time update in the KF.

        Parameters
        ----------
        x : numpy.ndarray
            The mean of the state estimate.
        P : numpy.ndarray
            The state error covariance.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            The updated mean and covariance.

        """
        xp = self.motion_model['f'](x)
        F = self.motion_model['F']
        Pp = F@P@F.T + self.motion_model['Q']
        return xp, Pp

    def update(self, x, P, res):
        """Measurement update in the KF.

        Modifies the estimates.

        Parameters
        ----------
        x : numpy.ndarray
            The mean of the state estimate.
        P : numpy.ndarray
            The state error covariance.
        res : numpy.ndarray
            The residual to use for the update.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            The updated mean and covariance.

        """
        H=np.array([[1, 0], \
                    [0, 1]]) # izvodi kao
        H = np.hstack([H, np.zeros((2, 2))])
        # print(P, H, R)
        Si = H@P@H.T + self.R
        Kk = P@H.T@np.linalg.inv(Si)
        xf = x+Kk@res # popravka estimacije na osnovu inovacije (res)
        Pf = P- Kk@Si@Kk.T
        return xf, Pf

class BasicTracker():
    def __init__(self, filt, clutter_model, gamma):
        """A basic single target tracker.

        Parameters
        ----------
        filt : filter
            See src.filter. Some sort of filter to use for the tracks.
        associator : associators
            See src.associators. Some sort of measurement associator (NN).
        gater : gater
            See src.gater. A gating function.
        clutter_model : dict
            A dict containing the clutter model.

        """
        self.filt = filt
        self.clutter_model = clutter_model
        self.gamma = gamma
        
    def gate(self, x, P, R, y):
        '''
        Mahalonobisova distanca.
        Odbacuje merenja koja nisu 'blizu' mete
        
        Parameters
        ----------
        mean : np.ndarray
               srednja vrednost pozicij ešpoikfd mete
        P : np.ndarray
            Kovarijaciona matrica greske merenja
        R: 
            kovarijaciona matrica za za senzor
        y : np.ndarray
            merenja
        gamma: 
             

        Returns
        -------
        Niz True/False da li se koristi merenje

        '''
        H=np.array([[1, 0], \
                    [0, 1]]) # izvodi kao
        H = np.hstack([H, np.zeros((2, 2))])
        Si= H @ P @ H.T + R
        accepted = []
        #print(y)
        
        if y.ndim < 2:
            y = np.expand_dims(y, 0)

            
        for j in range(y.shape[1]): # sva merenja u toj iteraciji
            #print(y[:,j], x)
            r = y[:,j]-x[:2]
            rastojanje_s = r.T @ np.linalg.inv(Si) @ r
            if rastojanje_s < self.gamma:
                accepted.append(True)
            else:
                accepted.append(False)
                
        accepted=np.array(accepted)
        return accepted
    
    def associate(self, eps):
        """A nearest neighbour associator."""
        """Find the minimal residual.

        Parameters
        ----------
        eps : numpy.ndarray
            Residuals.

        Returns
        -------
        int
            The index of the minimal residual.

        """
        r = np.linalg.norm(eps, axis=0)
        yind = np.argmin(r)
        return yind
    
    def evaluate(self, Y, x, P):
        """ Evaluates the measurements in Y.

        nx -- state dimension
        K -- number of measurements

        Parameters
        ----------
        Y : list
            List of detections at time k=0 to K where K is the length of Y.
            Each entry of Y is ny by N_k where N_k is time-varying as the number
            of detections vary.
        x : numpy.ndarray (nx x K)
            Array to save the state estimates in.
        P : numpy.ndarray (nx x nx x K)
            Array to save the state error covariances in.

        Returns
        -------
        numpy.ndarray (nx x K)
        
        
        ad
            State estimates per time.
        numpy.ndarray (nx x nx x K)
            State error covariance per time.
        """
        R=self.filt.R
        for k, meas_k in enumerate(Y):
            # Calculate prediction error of each measurement
            yhat = x[:2, k]
            eps = meas_k-yhat[:, None]
            # Gating step
            accepted_meas = self.gate(x[:, k], P[:, :, k], R, meas_k)
            # If any measurements are accepted, select the nearest one
            if accepted_meas.any():
                # Association step
                #print(eps, accepted_meas)
                yind = self.associate(eps[:, accepted_meas])
                # Update
                x[:, k], P[:, :, k] = self.filt.update(x[:, k], P[:, :, k], eps[:, accepted_meas][:, yind])
            # Propagate state estimate
            if k < len(Y)-1:
                x[:, k+1], P[:, :, k+1] = self.filt.propagate(x[:, k], P[:, :, k])
        return x, P
    
#%%
T = trajectories1()
Y=add_noise({'T': T['X4']}, PD, clutter_model, rng=np.random.default_rng(13))

path_plot(T, Y)
    
#%%

R = np.diag([1, 1])
PD = 0.9
lam = 2
volume = dict(xmin=-50, xmax=2500, ymin=-50, ymax=1100)
clutter_model = dict(volume=volume, lam=lam)

# CV model
Q = np.identity(2)*100
motion_model = cv_model(Q=Q, T=1)

# Setup Gater
gamma = 4.7

# Setup filter
filt = KF(motion_model, R)
# Setup tracker
tracker = BasicTracker(filt, clutter_model, gamma)
#result = evaluate_tracker(ex1_trajectories['T4'], tracker, 10)


N=len(Y)
print(N)

xhat = np.zeros((4, N))
xhat[:2, 0] = T['X4'][:, 0]
xhat[2:, 0] = T['X4'][:, 1]-T['X4'][:, 0]
Phat = np.zeros((4, 4, N))

results = tracker.evaluate(Y, xhat, Phat)

#%%
plt.figure()
plt.plot(results[0][0,:], results[0][1,:])
plt.show()

path_plot(T, Y, results[0])

