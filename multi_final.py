import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy

def trajectories1():
    """

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
    N2=round(r*np.pi/(100/3.6)) #recimo, treba prema brzini
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
    cov=np.array([[1, 0], 
                  [0, 1]])
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
        y=y.reshape((2, -1))
        Y.append(y)
        
    return Y

def path_plot(T, Y, confirmed_tracks=None):
    plt.figure()
    for key, X in T.items():
        # print(key, item.shape)
        plt.plot(X[0,:], X[1,:], 'g', linewidth=0.5)
        for i in range(len(Y)): 
            if Y[i].any():
                plt.scatter(Y[i][0, :], Y[i][1,:], marker='o', color='black', s=0.5)
    # plt.xlim([0, 2500])
    # plt.ylim([0, 1800])

    if not confirmed_tracks is None:
        legend=[]
        for i, ct in enumerate(confirmed_tracks):
            x=[]; y=[];
            for X in ct['x']:
                x.append(X[0])
                y.append(X[1])
            plt.plot(x, y, label=str(i+1))
            legend.append(str(i))
            
        plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
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
    def __init__(self, motion_model, R, PD):
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
        self.PD=PD

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

        Modifies the estimates in-place.

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
        #print(P, H, R)
        Si = H@P@H.T + self.R
        Kk = P@H.T@np.linalg.inv(Si)
        xf = x+Kk@res
        Pf = P- Kk@Si@Kk.T
        return xf, Pf

def score_logic(y, filt, state, params):
    """ Implements a score logic for tracks.

    Modifies everything in-place! Also returns the state (track).

    Parameters
    ----------
    y : numpy.ndarray
        The measurement to use for the logic. May be empty.
    filt : filter
        See src.filters. The filter used for the track.
    state : dict
        Fields:
            stage : tentative/confirmed/deleted
            Lt : track score
            x : list of state estimates
            P : list of state error covariances
    params : dict
        parameters to the score logic
        Fields:
            PD : probability of detection
            PG : probability of gating
            lam : forgetting factor for recursive score update
            Ptm : Probability of rejecting true track
            Pfc : Probability of confirming false track
            Bfa : False alarm rate
            Ldel : threshold to remove a confirmed track
    """
    if state['stage'] not in ['tentative', 'confirmed', 'deleted']:
        return state
    Lh = np.log((1-params['Ptm'])/params['Pfc'])
    Ll = np.log((params['Ptm'])/(1-params['Pfc']))
    if y.size != 0: # Measurement associated to track
        yhat = state['x'][-1][:2]
        H=np.array([[1, 0], \
                    [0, 1]]) # izvodi kao
        H = np.hstack([H, np.zeros((2, 2))])
        
        Sk = H@state['P'][-1]@H.T+filt.R
        yl = stats.multivariate_normal.pdf(y, mean=yhat, cov=Sk)
        lt = np.log(params['PD']*params['PG']*yl/params['Bfa'])
    else:
        lt = np.log(1-params['PD']*params['PG'])
    Lt = params['lam']*state['Lt']+lt
    if state['stage'] == 'tentative':
        if Lt >= Lh:
            state['stage'] = 'confirmed'
        elif Lt <= Ll:
            state['stage'] = 'deleted'
    if state['stage'] == 'confirmed':
        if Lt <= params['Ldel']:
            state['stage'] = 'deleted'
    state['Lt'] = Lt
    return state

def nm_logic(y, filt, state, params):
    """ Implements an N/M logic for tracks.

    Modifies everything in-place! Also returns the state (track).

    Parameters
    ----------
    y : numpy.ndarray
        The measurement to use for the logic. May be empty.
    filt : filter
        See src.filters. To ensure the logics have the same callsign, this is
        included.
    state : dict
        Fields:
            stage : tentative/confirmed/deleted
            nass : number of measurements associated to target
            nmeas : number of measurements possibly associated to target since birth
    params : dict
        parameters to the N/M logic
        N1/M1 - Number of required associations of the last M1 measurements for a stage one tentative track
        Fields:
            N1 : number of required associations to a stage one tentative track
            M1 : number of required measurements to a stage one tentative track
            N2 : number of required associations to a stage two tentative track
            M2 : number of required measurements to a stage two tentative track
            N3 : number of required missed measurements to delete a confirmed track

    """
    if state['stage'] not in ['tentative', 'confirmed', 'deleted']:
        return state
    print('state: ', state)
    state['nmeas'] += 1
    if y.size != 0:
        state['nass'] += 1
    if state['stage'] == 'tentative':
        if state['nmeas'] <= params['M1']:
            # Stage one tentative deletion
            if state['nass'] != state['nmeas']:
                state['stage'] = 'deleted'
        elif state['nmeas'] <= params['M1']+params['M2']:
            # Stage two tentative deletion
            if state['nmeas']-state['nass'] > params['M2']-params['N2']-(params['M1']-params['N1']):
                state['stage'] = 'deleted'
            elif state['nmeas'] == params['M1']+params['M2']:
                state['stage'] = 'confirmed'
                state['nmeas'] = 0
                state['nass'] = 0
    else: # Otherwise it is confirmed
        if y.size != 0: # Reset counters if association
            state['nass'] = 0
            state['nmeas'] = 0
        if state['nmeas'] - state['nass'] == params['N3']:
            state['stage'] = 'deleted'
            
    return state  
  

class GNN():
    def __init__(self, logic, logic_params, init_track, filt, gamma, clutter_model):
        """An implementation of a Global Nearest Neighbour tracker.

        Parameters
        ----------
        logic : logic
            See src.logic. Some sort of track logic.
        logic_params : dict
            Contains parameters to the track logic.
        init_track : callable
            A function that initiates a track. Should take a measurement, the
            time, an id and the filter to use for the track as input.
        filt : filter
            See src.filter. Some sort of filter to use for the tracks.
        gater : gater
            See src.gater. A gating function.
        clutter_model : dict
            A dict containing the clutter model.

        """
        self.logic = logic
        self.logic_params = logic_params
        self.init_track = init_track
        self.filt = filt
        #self.gater = gater
        self.gamma = gamma
        self.clutter_model = clutter_model
        
    def gate(self, x, P, R, y):
        '''
        Mahalonobisova distanca.
        Odbacuje merenja koja nisu 'blizu' mete
        
        Parameters
        ----------
        mean : np.ndarray
               srednja vrednost pozicije mete
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
    
    #######################################################
    
    def get_association_matrix(self, meas, tracks):
        """ Computes the validation and association matrix (specifically for the
            GNN implementation)

        Parameters
        ----------
        meas : numpy.ndarray
            Measurements to attempt to associate
        tracks : list
            A list of the tracks to associate the measurements with

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Returns an association matrix of size ny by Nc+2*ny and a validation
            matrix of size ny by Nc, where Nc is the number of tracks. The association
            matrix also contains the false alarm and new track possibilities.

        """
        
        ny = meas.shape[1]
        print('ny = ', ny)
        Nc = len(tracks) # Number of tracks to associate
        validation_matrix = np.zeros((ny, Nc), dtype=bool)

        association_matrix = -np.inf*np.ones((ny, Nc+2*ny))
        # Entry for false alarms
        np.fill_diagonal(association_matrix[:, Nc:Nc+ny], np.log(self.logic_params['Bfa']))
        # Entry for new targets
        np.fill_diagonal(association_matrix[:, Nc+ny:], np.log(self.logic_params['Bnt']))

        if tracks:
            # All of the tracks are assumed to use the same sensor model!
            x = np.vstack([track['x'][-1] for track in tracks]).T
            print(x.shape)
            # yhat_t = tracks[0]['filt'].sensor_model['h'](x) # Returns a (ny x nx) matrix
            yhat_t = x[:2, : ]
            # H_t = tracks[0]['filt'].sensor_model['dhdx'](x) # Returns a (ny x nC x nx x nC) tensor
            H=np.array([[1, 0], \
                        [0, 1]]) # izvodi kao
            H = np.hstack([H, np.zeros((2, 2))])
            
            for ti, track in enumerate(tracks): # Iterate over confirmed tracks
                validation_matrix[:, ti] = self.gate(track['x'][-1], track['P'][-1], track['filt'].R, meas)
                if validation_matrix[:, ti].any(): # If any measurements are validated
                    val_meas = meas[:, validation_matrix[:, ti]] # Get the validated measurements for this track
                    if Nc == 1: # Because of how numpy handles its matrices
                        yhat = yhat_t
                    else:
                        yhat = yhat_t[:, ti]
                    py = stats.multivariate_normal.pdf(val_meas.squeeze().T, mean=yhat.flatten(), cov=H@track['P'][-1]@H.T+track['filt'].R)
                    association_matrix[validation_matrix[:, ti], ti] = np.log(track['filt'].PD*py/(1-track['filt'].PD)) # PG assumed = 1
                    
        return association_matrix, validation_matrix

    def update_track(self, meas, track):
        """Handles the update of a certain track with the given measurement(s).

        Modifies the track in-place!

        Parameters
        ----------
        meas : numpy.ndarray
            Contains measurement(s) to update a specific track with. ny by N,
            where N is the number of measurements to update the track with.
        track : dict
            A dict containing everything relevant to the track.

        """
        if meas.size == 0:
            track = self.logic(np.array([]), track['filt'], track, self.logic_params) # If no meas associated, still update logic of track
            return
        
        # Calculate prediction error of each measurement
        yhat = track['x'][-1][:2]
        print('yhat = ', yhat)

        eps = meas-yhat
        track = self.logic(meas, track['filt'], track, self.logic_params)

        # Update
        track['x'][-1], track['P'][-1] = track['filt'].update(track['x'][-1], track['P'][-1], eps)

    def associate_update(self, meas_k, k, tracks, unused_meas):
        """Associates measurements to tracks and updates the tracks with the
        measurements.

        Does *not* return anything, but modifies the objects in-place! Uses
        scipy to compute the optimal assignment.

        Parameters
        ----------
        meas_k : numpy.ndarray
            Measurements to attempt to associate
        k : int
            The current time step
        tracks : list
            A list of the tracks to associate the measurements with
        unused_meas : numpy.ndarray
            A logical array indicating what measurements are still
            unused/non-associated.

        """
        association_matrix, validation_matrix = self.get_association_matrix(meas_k[:, unused_meas], tracks)
        # Solve association problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-association_matrix)
        for row, col in zip(row_ind, col_ind):
            if col >= len(tracks): # No target to associate the measurement to
                continue
            else:
                # Update confirmed tracks
                self.update_track(meas_k[:, unused_meas][:, row], tracks[col])
                tracks[col]['associations'].append(k) # If we've associated something, add the time here (for plotting purposes)
        for i in range(len(tracks)):
            if i not in col_ind:
                self.update_track(np.array([]), tracks[i])
        # Remove any gated measurements from further consideration
        tmp = unused_meas[unused_meas] # Extract the unused measurements
        inds = np.where(validation_matrix.sum(axis=1))
        tmp[inds] = 0
        unused_meas[unused_meas] = tmp

    def evaluate(self, Y):
        """ Evaluates the detections in Y.

        Parameters
        ----------
        Y : list
            List of detections at time k=0 to K where K is the length of Y.
            Each entry of Y is ny by N_k where N_k is time-varying as the number
            of detections vary.

        Returns
        -------
        list, list
            First list contains all initiated tracks, both tentative, deleted
            and confirmed. The second list contains only the confirmed list,
            even if they have died. Hence, the lists contain duplicates (but
            point to the same object!).

        """
        rng = np.random.default_rng()
        tracks = [] # Store all tracks
        confirmed_tracks = [] # Store the confirmed tracks (for plotting purposes only)
        ids = 0
        for k, meas_k in enumerate(Y):
            print('Merenje: ', k)
            ny = meas_k.shape[1]
            unused_meas = np.ones((ny), dtype=bool)

            # Handle the confirmed and alive tracks
            live_tracks = [track for track in confirmed_tracks if track['stage']=='confirmed']
            if live_tracks:
                self.associate_update(meas_k, k, live_tracks, unused_meas)

            # Handle the tentative tracks
            tentative_tracks = [track for track in tracks if track['stage'] == 'tentative']
            if tentative_tracks:
                self.associate_update(meas_k, k, tentative_tracks, unused_meas)
                for track in tentative_tracks:
                    if track['stage'] == 'confirmed':
                        confirmed_tracks.append(track) # If a track has been confirmed, add it to confirmed tracks

            # Use the unused measurements to initiate new tracks
            while unused_meas.any():
                ind = rng.choice(np.arange(unused_meas.size), p=unused_meas/unused_meas.sum())
                track = self.init_track(meas_k[:, ind], k, ids, self.filt) # Initialize track
                tracks.append(track)
                unused_meas[ind] = 0 # Remove measurement from association hypothesis
                self.associate_update(meas_k, k, [track], unused_meas)
                ids += 1
                if track['stage'] == 'confirmed':
                    confirmed_tracks.append(track)

            for track in tracks:
                if track['stage'] != 'deleted':
                    x, P = track['filt'].propagate(track['x'][-1], track['P'][-1])
                    track['x'].append(x)
                    track['P'].append(P)
                    track['t'].append(k+1)
        return tracks, confirmed_tracks

#%%
trajectories = trajectories2()
# trajs = ['X1', 'X3', 'X5', 'X6'] # Select the trajectories to use
trajs = ['X4', 'X5', 'X6'] # Select the trajectories to use

T = {key: T for key, T in trajectories.items() if key in trajs}


R = np.diag([10, 10])
PD = 0.9
lam = 2
volume = dict(xmin=-100, xmax=2500, ymin=0, ymax=1900)
V = (volume['xmax']-volume['xmin'])*(volume['ymax']-volume['ymin'])
Bfa = lam/V
clutter_model = dict(volume=volume, lam=lam)
Bnt = Bfa

Y=add_noise(T, PD, clutter_model, rng=np.random.default_rng(1))
path_plot(T, Y)
    
#%%

# CV model
Q = np.identity(2)*100
motion_model = cv_model(Q=Q, T=1)

# Setup Gater
gamma = 4.7

P0 = np.diag([10, 10, 1000, 1000])
Ptm = 0.01
Pfc = 0.001
lam = 0.6
logic_params = dict(PD=PD, PG=1, lam=lam, Ptm=Ptm, Pfc=Pfc,\
                    Bfa=Bfa, Ldel=1*np.log(Ptm/(1-Pfc)), Bnt=Bnt)

def init_track(y, k, identity, filt):
    track = dict(stage='tentative', Lt=0, x=[], P=[], t=[k], identity=identity,\
                 associations=[k], filt=filt)
    x0 = np.concatenate( y[:, None] ).flatten()
    x0 = np.hstack([x0, np.zeros((2,))])
    track['x'] = [x0]
    track['P'] = [P0]
    
    return track    

# Setup filter
filt = KF(motion_model, R, PD)

gnn = GNN(score_logic, logic_params, init_track, filt, gamma, clutter_model)
tracks, confirmed_tracks = gnn.evaluate(Y)

path_plot(T, Y, confirmed_tracks)

