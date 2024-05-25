from Func import *


# Start the timer
start_time = ti.time()

# For test IDS
# ---------------
# ID = [10875245]
# author = 'Kepler'
# Pl = ["K107"]
# Q = [2]
# PerExoRGB = [13.6]
# PerRef = [20.3]
# Per_uperrExoRGP = [2.6]
# Per_DnerrExoRGP = [1.5]
# Per_uperrRef = [3.3]
# Per_DnerrRef = [0.01]

# For All IDS
# -------------
# ID = [10875245,12302530,10619192,9478990,9818381,5794240,8435766,757450,5383248]
# Pl = ["K107","K155", "K17", "K39", "K43", "K45", "K78", "K75","K96"]
# Q = [2,5,1,3,1,1,1,2,4]
# PerExoRGB = [13.6,27.5,12,4.5,13.3,15.4,12.7,19.5,15.9]
# PerRef = [20.3,26.43,12.01,4.5,12.95,15.8,12.588,19.18,15.3]

# author = 'Kepler'
# Per_uperrExoRGP = [2.6,1.2,0.1,0.1,1,0.6,0.2,0.2,0.4]
# Per_DnerrExoRGP = [1.5,1,0.2,0.1,0.7,0.6,0.2,0.2,0.4]
# Per_uperrRef = [3.3,1.32,0.16,0.07,0.25,0.2,0.03,0.25,0.01]
# Per_DnerrRef = [0.01,1.32,0.16,0.07,0.25,0.2,0.03,0.25,0.01]
# ------------------------------------------------------------
# read from files
# KID, KOI, Pphot, Pspec, errPspec_Up, errPspec_Dn, 
# vsini, vsinierr, Rstar, errRstar_Up,errRstar_Dn, inclination

fname = './ref_per_matched_Q2_Q3_Q9.csv'
df = pd.read_csv(fname)
ID = df['KID']
Pl = df['KID']
Q = df['Q']
PerExoRGB = df['Pphot']
PerRef = df['Pspec']
author = 'Kepler'
Per_uperrExoRGP = df['errPspec_Up']
Per_DnerrExoRGP = df['errPspec_Dn']
Per_uperrRef = df['errPspec_Up']
Per_DnerrRef = df['errPspec_Dn']


# If you have ID
target_ids = ['KIC ' + str(id) for id in ID]  # Ensure target_ids is a list of strings

# target_ids = ID
for ii, target_id in enumerate(target_ids):
   print("start ------------------- ID:",target_id)
#    if ((target_id !='KIC 8866102')): 
   if (( Q[ii] != 4)):     
#     save_filepath = f'./{ID[ii]}.png'
    time_lc, flux_lc, time_s, flux_s, time_f, flux_f=plot_light_curves(target_id, author, Q[ii], window_length=51)
#     else:
#         QQ = None
#         time_lc, flux_lc, time_s, flux_s, time_f, flux_f=plot_light_curves(target_id, author, QQ, window_length=51)

    time, flux = time_s, flux_s

# For getting directly from file of online
#     filename1 = f'./out_data/{ID[ii]}.csv'
#     if os.path.exists(filename1):
#         df1 = pd.read_csv(filename1)
#         time = df1['Time']
#         flux = df1['Flux']
#     else:
#         time, flux = readID(target_id, quarter=Q[ii], filename=str(ID[ii]))

    valid_indices = np.isfinite(flux)
    time = time[valid_indices]
    flux = flux[valid_indices]
    lc = lk.LightCurve(time=time, flux=flux)
#     lc = lk.search_lightcurve(target_id).download().PDCSAP_FLUX
#     rmse_GP, rmse_RF, rmse_KN,rmse_XGB, rmse_Mu = model(time, flux)

    rmse_GP, rmse_RF, rmse_KN, rmse_XGB, rmse_DT, rmse_GB, rmse_weighted, rmse_ensemble,knn_score,RF_score,GB_score, DT_score,XGB_score,voting_pred_score,weighted_voting_pred_score,Voting_CM,Voting_FPR= model(time, flux)
#     ,voting_TPR,voting_TNR,voting_PPV,voting_NPV,voting_FPR,voting_FNR,voting_FDR 
    # RF_TPR,RF_TNR,RF_PPV,RF_NPV,RF_FPR,
    # RF_FNR,RF_FDR,KN_TPR,KN_TNR,KN_PPV,KN_NPV,KN_FPR,KN_FNR,KN_FDR,DT_TPR,DT_TNR,DT_PPV,DT_NPV,DT_FPR,DT_FNR,DT_FDR 

    time_RF, flux_RF = flux_prediction(time, flux, model='RandomForest')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_RF, flux_RF)
#     P_RF = ls_periodogram(time_RF, flux_RF, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_RF,err_RF = ls_periodogram(time_RF, flux_RF, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='RF')


    time_GP, flux_GP = flux_prediction(time, flux, model='Gaussian')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_GP, flux_GP)
#     P_GP = ls_periodogram(time_GP, flux_GP, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_GP,err_GP = ls_periodogram(time_GP, flux_GP, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='GP')

    time_KN, flux_KN = flux_prediction(time, flux, model='KNeighbors')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_KN, flux_KN)
#     P_KN = ls_periodogram(time_KN, flux_KN, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_KN,err_KN = ls_periodogram(time_KN, flux_KN, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='KN')

    time_XGB, flux_XGB = flux_prediction(time, flux, model='XGBRegressor')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_XGB, flux_XGB)
#     P_XGB = ls_periodogram(time_XGB, flux_XGB, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_XGB,err_XGB = ls_periodogram(time_XGB, flux_XGB, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='XGB')


    time_DT, flux_DT = flux_prediction(time, flux, model='DecisionTree')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_DT, flux_DT)
#     P_DT = ls_periodogram(time_DT, flux_DT, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_DT,err_DT = ls_periodogram(time_DT, flux_DT, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='DT')
    

    time_GB, flux_GB = flux_prediction(time, flux, model='GradientBoosting')
    bls_period, bls_t0, bls_depth = BLS(lc.flatten(), time_GB, flux_GB)
#     P_GB = ls_periodogram(time_GB, flux_GB, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period)
    P_GB,err_GB = ls_periodogram(time_GB, flux_GB, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='GB')
    

    time_Mu, flux_Mu = flux_prediction(time, flux, model='MultiVote')
    bls_period_Mu, bls_t0, bls_depth = BLS(lc.flatten(), time_Mu, flux_Mu)
#     P_Mu = ls_periodogram(time_Mu, flux_Mu, xlim_min=1, xlim_max=time_Mu.max()-time_Mu.min(), bls=bls_period_Mu)
    P_Mu,err_Mu = ls_periodogram(time_Mu, flux_Mu, xlim_min=1, xlim_max=time.max()-time.min(), bls=bls_period,target_id=target_id,Model='Mu')
    print('P_Ref:',PerRef[ii],'PerExoRGB',PerExoRGB[ii], 'P_Mu:',P_Mu, 'P_XGB:',P_XGB, 'P_RF:',P_RF,'P_GB:',P_GB, 'P_DT:',P_DT )

    # Save rmse values in a CSV file
    rmse_data = pd.DataFrame({
        'Pl':[Pl[ii]],
        'ID': [ID[ii]],
#         'Kp': [Kp[ii]],
#         'Rad': [Rad[ii]],
        'PerExoRGB': [PerExoRGB[ii]],
        'PerRef': [PerRef[ii]],
        'Per_uperrExoRGP': [Per_uperrExoRGP[ii]],
        'Per_DnerrExoRGP': [Per_DnerrExoRGP[ii]],
        'Per_uperrRef': [Per_uperrRef[ii]],
        'Per_DnerrRef': [Per_DnerrRef[ii]],
#         'R_star': [R_star[ii]],
        'rmse_DT': [rmse_DT],
        'rmse_RF': [rmse_RF],
        'rmse_KN': [rmse_KN],
        'rmse_GB': [rmse_GB],
        'rmse_XGB': [rmse_XGB],
        'rmse_Mu': [rmse_ensemble],
        'knn_score': [knn_score],
        'RF_score': [RF_score],
        'RF_score': [RF_score],
        'GB_score': [GB_score],
        'DT_score': [DT_score],
        'XGB_score': [XGB_score],
        'P_RF': [P_RF],
        'P_DT': [P_DT],
        'P_KN': [P_KN],
        'P_GB': [P_GB],
        'P_GB': [P_GB],
        'P_Mu': [P_Mu],
        'err_RF': [err_RF],
        'err_DT': [err_DT],
        'err_KN': [err_KN],
        'err_GB': [err_GB],
        'err_XGB':[err_GB],
        'err_Mu': [err_Mu],          
        'Voting_CM' : [Voting_CM],
        'voting_TPR': [Voting_FPR[0]],
        'voting_TNR': [Voting_FPR[1]],
        'voting_PPV': [Voting_FPR[2]],
        'voting_NPV': [Voting_FPR[3]],
        'voting_FPR': [Voting_FPR[4]],
        'voting_FNR': [Voting_FPR[5]],
        'voting_FDR': [Voting_FPR[6]],
        'voting_ACC': [Voting_FPR[7]]
    })
    
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time
    date_string = current_datetime.strftime("%Y-%m-%d")
    time_string = current_datetime.strftime("%H-%M-%S")

    # Construct the new filename
    rmse_filename = f'./out_data/RMSE_{date_string}.csv'

    # rmse_filename = f'./out_data/RMSE.csv'

    if os.path.exists(rmse_filename):
        rmse_data.to_csv(rmse_filename, mode='a', header=False, index=False)
    else:
        rmse_data.to_csv(rmse_filename, index=False)

#     Plot light Curve
#     plot_LC(target_id) # No need for all IDs
#     plot models and averages include RMSE and ID
    plot_models(time,flux,  time_Mu, flux_Mu,time_KN, flux_KN, time_GB, flux_GB, time_XGB, flux_XGB, time_DT, flux_DT, time_RF, flux_RF, rmse_GB, rmse_XGB, rmse_RF, rmse_KN, rmse_DT,rmse_ensemble,target_id)

# End the timer
end_time = ti.time()

# Calculate the duration
duration = end_time - start_time

# Print the duration
print("Run duration:", duration, "seconds")