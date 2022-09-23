import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
import csv
import cv2
import time

def get_measurement(address, frame_dim, N_frames=None):
    Y=[]
    Rows_all=[]
    with open(address, 'r') as file:
        X=np.empty(shape=(2, 0))
        # za svaki frejm zapamti np.ndarray sredine box-ova
        frame=0
        Rows=[]
        for i, row1 in enumerate(csv.DictReader(file)):
            
            row=dict()
            for key, value in row1.items():
                if key=='conf':
                    row[key]=float(value)
                else:
                    row[key]=int(value)  
            
            if i==0:
                frame=int(row['frame'])
            if (not N_frames is None) and (int(row['frame'])>N_frames):
                return Y, Rows_all
            # print(row)

            x = np.array([[int(row['bb_left'])+int(row['bb_width'])/2],\
                          [frame_dim['height']-(int(row['bb_top'])+int(row['bb_height'])/2)]])
            if int(row['frame']) == frame:  
                X = np.concatenate((X, x), axis=1)
                Rows.append(row)
            else: 
                Y.append(X)
                Rows_all.append(Rows);
                Rows=[row]
                X=x
                frame=int(row['frame'])
        
    return Y, Rows_all
    
def plot_all(Y, frame_dim, confirmed_tracks=None):
    plt.figure()
    plt.xlim([0, frame_dim['width']])
    plt.ylim([0, frame_dim['height']])
    for i in range(len(Y)): 
        if Y[i].any():
            plt.scatter(Y[i][0, :], Y[i][1,:], marker='o', color='black', s=0.5)

    if not confirmed_tracks is None:
        # legend=[]
        for i, ct in enumerate(confirmed_tracks):
            x=[]; y=[];
            for X in ct['x']:
                x.append(X[0])
                y.append(X[1])
            plt.plot(x, y, label=str(ct['identity']))

    if len(Y)<200:
        plt.legend()
    plt.xlim([0, frame_dim['width']])
    plt.ylim([0, frame_dim['height']])
    plt.show()

def plot_some_path(frame_dim, confirmed_tracks=None, numbers_id=[0, 1, 2]):
    plt.figure()
    plt.xlim([0, frame_dim['width']])
    plt.ylim([0, frame_dim['height']])

    if not confirmed_tracks is None:
        # legend=[]
        for i, ct in enumerate(confirmed_tracks):
            x=[]; y=[];
            for X in ct['x']:
                x.append(X[0])
                y.append(X[1])
            if ct['identity'] in numbers_id:
                plt.plot(x, y, label=str(ct['identity']))


    plt.legend()
    plt.xlim([0, frame_dim['width']])
    plt.ylim([0, frame_dim['height']])
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
        The filter used for the track.
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
  

class GNN():
    def __init__(self, logic, logic_params, init_track, filt, gamma, clutter_model):
        """An implementation of a Global Nearest Neighbour tracker.

        Parameters
        ----------
        logic : logic
            track logic.
        logic_params : dict
            Contains parameters to the track logic.
        init_track : callable
            A function that initiates a track. Should take a measurement, the
            time, an id and the filter to use for the track as input.
        filt : filter
            filter to use for the tracks.
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
        # print('ny = ', ny)
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
            # print(x.shape)
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

        eps = meas-yhat
        track = self.logic(meas, track['filt'], track, self.logic_params)

        # Update
        track['x'][-1], track['P'][-1] = track['filt'].update(track['x'][-1], track['P'][-1], eps)

    def associate_update(self, orig_indices, meas_k, k, tracks, unused_meas):
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
           
                # print(unused_meas, 'koje', row)
                # pronadji koje merenje po redu je izabrano
                j=-1
                for br, u in enumerate(unused_meas):
                    if u:
                        j+=1
                    if j==row:
                        tracks[col]['which'].append(br) # to know from which measurement 
                        break
                
                
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
        t=[]
        for k, meas_k in enumerate(Y):
            start=time.time()
            print('Merenje: ', k)
            ny = meas_k.shape[1]
            unused_meas = np.ones((ny), dtype=bool)
            orig_indices = np.arange(0, ny)
            # print(unused_meas)
            # Handle the confirmed and alive tracks
            live_tracks = [track for track in confirmed_tracks if track['stage']=='confirmed']
            if live_tracks:
                self.associate_update(orig_indices, meas_k, k, live_tracks, unused_meas)

            # Handle the tentative tracks
            tentative_tracks = [track for track in tracks if track['stage'] == 'tentative']
            if tentative_tracks:
                self.associate_update(orig_indices, meas_k, k, tentative_tracks, unused_meas)
                for track in tentative_tracks:
                    if track['stage'] == 'confirmed':
                        confirmed_tracks.append(track) # If a track has been confirmed, add it to confirmed tracks

            # Use the unused measurements to initiate new tracks
            while unused_meas.any():
                # print(unused_meas)
                ind = rng.choice(np.arange(unused_meas.size), p=unused_meas/unused_meas.sum())
                track = self.init_track(ind, meas_k[:, ind], k, ids, self.filt) # Initialize track
                tracks.append(track)
                unused_meas[ind] = 0 # Remove measurement from association hypothesis
                # orig_indices.remove(ind)
                self.associate_update(orig_indices, meas_k, k, [track], unused_meas)
                ids += 1
                if track['stage'] == 'confirmed':
                    confirmed_tracks.append(track)

            for track in tracks:
                if track['stage'] != 'deleted':
                    x, P = track['filt'].propagate(track['x'][-1], track['P'][-1])
                    track['x'].append(x)
                    track['P'].append(P)
                    track['t'].append(k+1)
            
            end=time.time()
            t.append(end-start)
        return tracks, confirmed_tracks, t

#%% Ulazni fajlovi

in_file='video1.mp4'
detections_file='export1.csv'
#%% Pristupanje ulaznim podacima i postavka parametara za GNN

cam = cv2.VideoCapture(in_file)
# o ulaznom snimku
ret, frame = cam.read()

	
# cv2.imwrite('test.png', frame) 

frame_dim=dict()
frame_dim['height'], frame_dim['width'], layers = frame.shape
fps = cam.get(cv2.CAP_PROP_FPS)
T=1/fps;
cam.release()

# paramteri za senzor 
R = np.diag([10, 10])
PD = 0.9
lam = 1

volume = dict(xmin=0, xmax=frame_dim['width'], ymin=0, ymax=frame_dim['height'])
V = (volume['xmax']-volume['xmin'])*(volume['ymax']-volume['ymin'])
Bfa = lam/V
clutter_model = dict(volume=volume, lam=lam) # model za clutter prema veliini frejma
Bnt = Bfa

# formiranje merenje iz .csv fajla dobijenog iz NM
# N_frames=150
Y, Rows_all = get_measurement(detections_file, frame_dim, N_frames=None)
# print('Samo jos crtanje!')
# plot_all(Y, frame_dim)

# CV model
Q = np.identity(2)*100
motion_model = cv_model(Q=Q, T=T)

# Gama parametar za gating
gamma = 4.7

# Postavka parametara za logiku putanja
P0 = np.diag([100, 100, 1000, 1000])
Ptm = 0.01
Pfc = 0.001
lam = 0.6
logic_params = dict(PD=PD, PG=1, lam=lam, Ptm=Ptm, Pfc=Pfc,\
                    Bfa=Bfa, Ldel=1*np.log(Ptm/(1-Pfc)), Bnt=Bnt)


def init_track(ind, y, k, identity, filt):
    '''
    Funkcija za incijalizaciju nove putanje.

    Parameters
    ----------
    ind : int
        Koje merenje po redu inicira ovu putanju.
    y : np.ndarray
        Merenje koje inicira ovu putanju.
    k : int
        Trenutni frejm.
    identity : int
        ID nove putanje.
    filt : object
        Filter (Kalmanov) koji se koristi za update putanje kasnije.

    Returns
    -------
    track : dict
        Putanja sa svim potrebnim poljima.
            'stage': 'confirmed'/'tentative'/'deleted'
            'which': (list) sadrzi koja merenje cine tu putanju
            
    '''
    track = dict(stage='tentative', Lt=0, x=[], P=[], t=[k], identity=identity,\
                 associations=[k], filt=filt, which=[ind])
    
    x0 = np.concatenate( y[:, None] ).flatten()
    x0 = np.hstack([x0, np.zeros((2,))])
    track['x'] = [x0]
    track['P'] = [P0]
    
    return track 
    
# Postavka filtra, koristi se Kalmanov filtar
filt = KF(motion_model, R, PD)

# postavka GNN-a (Global Nearest Neighbour)
gnn = GNN(score_logic, logic_params, init_track, filt, gamma, clutter_model)
tracks, confirmed_tracks, t = gnn.evaluate(Y)

print('Samo jos crtanje!')       
plot_all(Y, frame_dim, confirmed_tracks)
#%%

plt.figure()
plt.hist(t, bins=50)
plt.xlabel('Vreme izvršavanja po frejmu')
plt.ylabel('Broj frejmova')
plt.show()

#%%

plot_some_path(frame_dim, confirmed_tracks, numbers_id=[0, 5, 10, 7])

#%% Pravljenje novog snimka sa ID-jevima

def make_video(in_file, out_file, Y, confirmed_tracks):
    '''
    Ucitava detekcije njihove ID-jeve na svakom frejmu ulaznog snimka.

    Parameters
    ----------
    in_file : string
        Ulazni snimak
    out_file : string
        Izlazni snimak sa svim detekcijama

    Returns
    -------
    None.

    '''
    # which
    # Sadrzi informacije o tome koje merenje pripada kojoj putanji;
    # ako ne pripada ni jednoj polje ima vrednost -1.
    which=[[]]*len(Y)
    for i, y in enumerate(Y):
        which[i]=np.ones((y.shape[1], ), dtype=int)*-1
        
    max_identity=0
    for ct in confirmed_tracks:
        for i, t in enumerate(ct['associations']): 
            which[t][ct['which'][i]] = ct['identity']
            if ct['identity']>max_identity:
                max_identity=ct['identity']
    
    
    #######################################################
    cam = cv2.VideoCapture(in_file)
    
    # o ulaznom snimku
    ret, frame = cam.read()
    height, width, layers = frame.shape
    size = (width, height)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print('FPS = ', fps)
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    caption_font = cv2.FONT_HERSHEY_TRIPLEX # FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN 
    color = [np.random.random(size=3) * 256 for i in range(max_identity+1)]
    caption_size = 0.5
    caption_thickness = 1
    
    i=0 # frame number
    while (ret):
        # dodaj detekcije na slici po identity-ju iz confirmed_tracks (which)
        # za i-ti frejm
        img=frame.copy()
        if i>=len(Rows_all):
            break
        Rows= Rows_all[i]
        
        
        for j, row in enumerate(Rows):
            print(i, j)
            # print(row)
            top_left = row['bb_left'], row['bb_top']
            bottom_right = row['bb_left']+row['bb_width'], row['bb_top']+row['bb_height']
            # print(top_left, bottom_right)
            
            if which[i][j]!=-1:
                # generate bbox's caption
                caption = str(int(which[i][j]))
                (w, h), _ = cv2.getTextSize(caption, caption_font, caption_size, caption_thickness)
                caption_left = row['bb_left'],  row['bb_top']
                # caption_right = row['bb_left'] + w, row['bb_top']+row['bb_height']+h
                # cv2.rectangle(img, caption_left, caption_right, (0,0,0), -1 )
                cv2.rectangle(img, top_left, bottom_right, color[which[i][j]], 1)
                cv2.putText(img, caption, caption_left, caption_font,  caption_size, color[which[i][j]], caption_thickness)
            else:
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)


        img=img.astype(np.uint8)
        out.write(img)

        ret, frame = cam.read()
        i+=1
    
    # Release all space and windows once done
    cam.release()
    out.release()

#%%
print(in_file)
out_file='out2.mp4'
make_video(in_file, out_file, Y, confirmed_tracks)