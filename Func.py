import datetime
import lightkurve as lk
import numpy as np
import time as ti
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor, XGBClassifier
# from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier  
from astropy.timeseries import BoxLeastSquares
# from docx import Document
# from io import BytesIO
# import exoplanet as xo
from astropy.timeseries import LombScargle
from estimators import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.cluster import KMeans
visible_ticks = {"top": True, "right": True}
font_style = {'fontname': 'Times New Roman', 'fontsize': 15}
# import os
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# import lightkurve as lk


# our code
# ----------------------------------------------------


def plot_light_curves(target_ids, author, quarter, window_length=401, max_retries=100):
    
 retry_count = 0
 valid_time = None
 valid_flux = None
    
 while retry_count < max_retries:
  try:    
    lc = lk.search_lightcurve(target_ids, author=author, quarter=quarter).download()
    clean_lc = lc.remove_outliers().flatten(window_length=window_length)
    flat, trend = lc.flatten(window_length=window_length, return_trend=True)

    time_lc = lc.time.value
    flux_lc = lc.flux.value
    # Given data
    time = flat.time.value
    flux = flat.flux.value

    time_s = trend.time.value
    flux_s = trend.flux.value

    # Remove NaN values
    valid_mask = np.isfinite(flux)
    time_valid = time[valid_mask]
    flux_valid = flux[valid_mask]

    # Reshape the data
    time_valid = time_valid.reshape(-1, 1)
    flux_valid = flux_valid.reshape(-1, 1)

    # Create and fit the random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = GaussianProcessRegressor()
    model.fit(time_valid, flux_valid)

    # Generate points for prediction
    time_fit = np.linspace(min(time_valid), max(time_valid), 10000).reshape(-1, 1)

    # Predict flux for the fitted time values
    flux_fit = model.predict(time_fit)

    fig, axs = plt.subplots(1,3, figsize=(15,3), sharex=False)

    # Plot main light curve
    axs[0].plot(time_lc, flux_lc, label='Main Light Curve ' + str(target_ids), color='black', lw=1)
    axs[0].set_ylabel('Flux')
    axs[0].set_xlabel('Time')
    axs[0].legend(loc = 'lower right')

    # Plot stellar activity (rotation)
    axs[1].plot(time_s, flux_s, 'b-', label='Stellar Activity (Rotation)', lw=1)
    axs[1].set_ylabel('Flux')
    axs[1].set_xlabel('Time')
    axs[1].legend(loc = 'lower right')

    # Plot flattened light curve
    # axs[2].plot(time_fit, flux_fit, label='Flattened Light Curve', color='royalblue', lw=1)
    axs[2].plot(time, flux, label='Flattened Light Curve', color='royalblue', lw=1)

    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Normalized Flux')
    axs[2].legend(loc = 'lower right')

    plt.tight_layout()
    
    # Create the 'output/plot' directory if it doesn't exist
    os.makedirs('output/plot', exist_ok=True)
    
    # Save the plot in 'output/plot' directory
    plt.savefig('output/plot/light_curve_'+str(target_ids)+'.png')
    
    #plt.show()

    return time_lc, flux_lc, time_s, flux_s, time, flux
  except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Connection error occurred: {e}")
            retry_count += 1
            print(f"Retrying... (Attempt {retry_count}/{max_retries})")
            time.sleep(5)  # Wait for 5 seconds before retrying
    
  print(f"Max retries exceeded. Unable to retrieve data for Kepler ID: {KeplerID}")
  return None    

def readID(KeplerID, quarter=None, max_retries=100, filename= None):
    output = './output/'
    target_id = KeplerID
    retry_count = 0
    valid_time = None
    valid_flux = None
    
    while retry_count < max_retries:
        try:
            if quarter is None:
                # Download and process the light curve data
                lc = lk.search_lightcurve(target_id).download().PDCSAP_FLUX
            else:
                # Download and process the light curve data
                lc = lk.search_lightcurve(target_id, quarter=quarter).download().PDCSAP_FLUX
            
            # Convert time to days
            time_vals = lc.time.value
            time_vals -= np.min(time_vals)
            
            # Remove invalid data points
            valid_indices = np.isfinite(lc.flux) & np.isfinite(time_vals)
            valid_time = time_vals[valid_indices]
            valid_flux = lc.flux[valid_indices]
            
            # Create a DataFrame
            data = pd.DataFrame({'Time': valid_time, 'Flux': valid_flux})
            
            # Save DataFrame as CSV with Kepler ID as filename
            if filename:
                data.to_csv(output+f'{filename}.csv', index=False)
            
            # Download the first result

            return valid_time, valid_flux
        
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Connection error occurred: {e}")
            retry_count += 1
            print(f"Retrying... (Attempt {retry_count}/{max_retries})")
            time.sleep(5)  # Wait for 5 seconds before retrying
    
    print(f"Max retries exceeded. Unable to retrieve data for Kepler ID: {KeplerID}")
    return None


# Define the add_black_border function (place it before its first use)
def add_black_border(ax):
    # Set black color for all spines
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Set black tick parameters
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Set white background for the plot
    ax.set_facecolor('white')

def plot_LC(target_id):
        # Plot the light curve
    search_result = lk.search_lightcurve(target_id, mission='Kepler')

    # Download the first result
    lc_collection = search_result.download_all()

    # Plot the light curve
    if lc_collection:
        lc_collection[0].plot()
    plt.savefig(f'./output/{target_id}.png')   
    #plt.show()


def plot_models(valid_time, valid_flux, time_Mu, flux_Mu, time_KN, flux_KN, time_GB, flux_GB, time_XGB, flux_XGB, time_DT, flux_DT, time_RF, flux_RF, rmse_GB, rmse_XGB, rmse_RF, rmse_KN, rmse_DT,rmse_Mu, ID):
    # Sort the K-Nearest Neighbors data
    knn_sorted_indices = np.argsort(time_KN)
    time_KN_sorted = time_KN[knn_sorted_indices]
    flux_KN_sorted = flux_KN[knn_sorted_indices]

    # Sort the GradientBoosting
    gb_sorted_indices = np.argsort(time_GB)
    time_GB_sorted = time_GB[gb_sorted_indices]
    flux_GB_sorted = flux_GB[gb_sorted_indices]

    # Sort the GradientBoosting
    xgb_sorted_indices = np.argsort(time_XGB)
    time_XGB_sorted = time_XGB[xgb_sorted_indices]
    flux_XGB_sorted = flux_XGB[xgb_sorted_indices]

    # Sort the DecisionTree data
    dt_sorted_indices = np.argsort(time_DT)
    time_DT_sorted = time_DT[dt_sorted_indices]
    flux_DT_sorted = flux_DT[dt_sorted_indices]

    # Sort the Random Forest data
    rf_sorted_indices = np.argsort(time_RF)
    time_RF_sorted = time_RF[rf_sorted_indices]
    flux_RF_sorted = flux_RF[rf_sorted_indices]

    # Sort the Random Forest data
    mm_sorted_indices = np.argsort(time_Mu)
    time_Mu_sorted = time_Mu[mm_sorted_indices]
    flux_Mu_sorted = flux_Mu[mm_sorted_indices]

    
    # Calculate the average flux
    avg_flux = np.mean([flux_KN_sorted, flux_GB_sorted, flux_DT_sorted, flux_RF_sorted, flux_XGB_sorted], axis=0)
    
    # Create subplots for each model
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True, sharey=True)

    # Plot K-Nearest Neighbors Regression
    ax1 = axes[0, 0]
    ax1.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax1.plot(time_KN_sorted, flux_KN_sorted, color='purple', lw=1, label='KN (RMSE: {:.3f})'.format(rmse_KN))
    ax1.set_title('K-Nearest Neighbors Regression',**font_style)
#     ax1.set_xlabel('Time')
    ax1.set_ylabel('Flux',**font_style)
    leg1 = ax1.legend(loc='lower left')
    for t in leg1.get_texts():
        t.set_fontsize('x-large')
    # Plot Gaussian Process Regression
    ax2 = axes[0, 1]
    ax2.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax2.plot(time_GB_sorted, flux_GB_sorted, color='cyan', lw=1, label='GB (RMSE: {:.3f})'.format(rmse_GB))
    ax2.set_title('Gradient Boosting Regression',**font_style)
#     ax2.set_xlabel('Time')
    # ax2.set_ylabel('Flux')
    leg2 = ax2.legend(loc='lower left')
    for t in leg2.get_texts():
        t.set_fontsize('x-large')

    # Plot XGBoost Regression
    ax3 = axes[1, 0]
    ax3.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax3.plot(time_DT_sorted, flux_DT_sorted, color='orange', lw=1, label='DT (RMSE: {:.3f})'.format(rmse_DT))
    ax3.set_title('Decision Tree Regression',**font_style)
#     ax3.set_xlabel('Time')
    ax3.set_ylabel('Flux',**font_style)
    ax3.legend(loc='lower left')
    leg3 = ax3.legend(loc='lower left')
    for t in leg3.get_texts():
        t.set_fontsize('x-large')    
        # Plot Random Forest Regression
    ax4 = axes[1, 1]
    ax4.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax4.plot(time_RF_sorted, flux_RF_sorted, color='blue', lw=1, label='RF (RMSE: {:.3f})'.format(rmse_RF))
    ax4.set_title('Random Forest Regression',**font_style)
#     ax4.set_xlabel('Time')
    # ax4.set_ylabel('Flux')
    ax4.legend(loc='lower left')
    leg4 = ax4.legend(loc='lower left')
    for t in leg4.get_texts():
        t.set_fontsize('x-large')
       # Plot XGBoost Regression
    ax5 = axes[2, 0]
    ax5.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax5.plot(time_XGB_sorted, flux_XGB_sorted, color='green', lw=1, label='RF (RMSE: {:.3f})'.format(rmse_XGB))
    ax5.set_title('XGBoost Regression',**font_style)
    ax5.set_xlabel('Time',**font_style)
    ax5.set_ylabel('Flux',**font_style)
    ax5.legend(loc='lower left')
    leg5 = ax5.legend(loc='lower left')
    for t in leg5.get_texts():
        t.set_fontsize('x-large')
        # Plot MM
    ax6 = axes[2, 1]
    ax6.scatter(valid_time, valid_flux, color='grey',s=2, alpha=0.5, label=ID)
    ax6.plot(time_Mu_sorted, flux_Mu_sorted, color='red', lw=1, label='MM (RMSE: {:.3f})'.format(rmse_Mu))
    ax6.set_title('Voting Classifier (Multi-Model)',**font_style)
    ax6.set_xlabel('Time',**font_style)
    # ax6.set_ylabel('Flux')
    ax6.legend(loc='lower left')
    leg6 = ax6.legend(loc='lower left')
    for t in leg6.get_texts():
        t.set_fontsize('x-large')
    # Add new axes for average flux
    # ax5 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # ax5.scatter(valid_time, valid_flux, color='blue',s=2, alpha=0.7, label=ID)
    # ax5.plot(time_RF_sorted, avg_flux, color='orange', lw=2, label='MM (RMSE: {:.3f})'.format(rmse_Mu))
    # ax5.set_title('Vote Multi-model')
    # ax5.set_xlabel('Time')
    # ax5.set_ylabel('Multi-Model Flux')
    # ax5.legend(loc='lower left')

    # Adjust layout and display the plot
    plt.subplots_adjust(wspace=0.1,hspace=0.35, bottom=0.25, right=0.96,left=0.3, top=0.93)
    plt.tight_layout()
    plt.savefig(f'./output/models_{ID}.pdf')
    # #plt.show()




def PeriodID(KeplerID, quarter=None):

    target_ids = KeplerID  # Ensure target_ids is a list of strings

    for target_id in target_ids:
        valid_time, valid_flux = readID(target_id, quarter=quarter)
    
        if valid_time is not None and valid_flux is not None:
            lc = lk.LightCurve(time=valid_time, flux=valid_flux)
            periodogram = lc.to_periodogram()
            periods = periodogram.period
            periodogram.plot()
            max_power = periodogram.power.max()
            best_period = periods[periodogram.power.argmax()]
            plt.text(0.5, 0.9, f'Period: {best_period:.2f} days', horizontalalignment='center',
                     verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
            
            #plt.show()
            
        else:
            print(f"No light curve data found for target ID: {target_id}")
def calculate_false_positive_rate(true_labels, predicted_labels):
    CM = confusion_matrix(true_labels, predicted_labels) #.ravel()
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # return ACC,FPR
    return TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC

def model(time, flux):
    # Convert time to days
    time -= np.min(time)
    
    # Remove invalid data points
    valid_indices = np.isfinite(flux)
    valid_time = time[valid_indices]
    valid_flux = flux[valid_indices]
    
    # Additional features
    relative_time = valid_time / np.max(valid_time)  # Relative time feature
    
    # Split the data into training and test sets
    X = np.column_stack((valid_time, relative_time))
    y = valid_flux
    
    # Filter out invalid labels for XGBoost
    valid_labels = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[valid_labels]
    y = y[valid_labels]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the models Regressor -----------------------------------------------------
    print("Initialize the models Regressor-------------------")
    gaussian_model = GaussianProcessRegressor()
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    knn_model = KNeighborsRegressor()
    xgboost_model = XGBRegressor()

    # added new model
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)


    # Create a pipeline with scaling and regression
    pipeline_gaussian = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=gaussian_model,
                                                                              transformer=StandardScaler()))
    pipeline_random_forest = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=random_forest_model,
                                                                                    transformer=StandardScaler()))
    pipeline_knn = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=knn_model,
                                                                          transformer=StandardScaler()))
    pipeline_xgboost = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=xgboost_model,
                                                                              transformer=StandardScaler()))
    pipeline_decision_tree = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=decision_tree_model,
                                                                                  transformer=StandardScaler()))
    pipeline_gradient_boosting = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=gradient_boosting_model,
                                                                                        transformer=StandardScaler()))

    # Fit the pipelines on the training data
    pipeline_gaussian.fit(X_train, y_train)
    pipeline_random_forest.fit(X_train, y_train)
    pipeline_knn.fit(X_train, y_train)
    pipeline_xgboost.fit(X_train, y_train)
    pipeline_decision_tree.fit(X_train, y_train)
    pipeline_gradient_boosting.fit(X_train, y_train)


    # Predict on the test data
    flux_pred_gaussian = pipeline_gaussian.predict(X_test)
    flux_pred_random_forest = pipeline_random_forest.predict(X_test)
    flux_pred_knn = pipeline_knn.predict(X_test)
    flux_pred_xgboost = pipeline_xgboost.predict(X_test)
    flux_pred_decision_tree = pipeline_decision_tree.predict(X_test)
    flux_pred_gradient_boosting = pipeline_gradient_boosting.predict(X_test)


    # Calculate RMSE for each model
    rmse_gaussian = np.sqrt(mean_squared_error(y_test, flux_pred_gaussian))
    rmse_random_forest = np.sqrt(mean_squared_error(y_test, flux_pred_random_forest))
    rmse_knn = np.sqrt(mean_squared_error(y_test, flux_pred_knn))
    rmse_xgboost = np.sqrt(mean_squared_error(y_test, flux_pred_xgboost))
    rmse_decision_tree = np.sqrt(mean_squared_error(y_test, flux_pred_decision_tree))
    rmse_gradient_boosting  = np.sqrt(mean_squared_error(y_test, flux_pred_gradient_boosting ))

    # Calculate weights for each model
    weights = np.array([rmse_random_forest, rmse_knn, rmse_decision_tree, rmse_gradient_boosting,rmse_xgboost])
    weights = 1 / weights
    weights /= np.sum(weights)

    # Weighted prediction
    weighted_prediction = (weights[0] * flux_pred_random_forest +
                           weights[1] * flux_pred_knn +
                           weights[2] * flux_pred_decision_tree +
                           weights[3] * flux_pred_gradient_boosting +
                           weights[4] * flux_pred_xgboost)

    rmse_weighted = np.sqrt(mean_squared_error(y_test, weighted_prediction))

    # Create the ensemble model with voting
    ensemble_model = VotingRegressor([
        ('random_forest', pipeline_random_forest),
        ('knn', pipeline_knn),
        ('decision_tree', pipeline_decision_tree),
        # ('gradient_boosting', pipeline_gradient_boosting),
        # ('xgboost', pipeline_xgboost)
    ])
    # Impute missing values in the features using median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # Fit the ensemble model on the training data
    ensemble_model.fit(X_train_imputed, y_train)

    # Predict on the test data
    y_pred_ensemble = ensemble_model.predict(X_test_imputed)

    # Calculate RMSE for the ensemble model
    rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))

    # model_classifier -------------------------------------------------
    print("model_classifier  -------------------")
    # Initialize classifiers for classification
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    decision_tree_classifier = GradientBoostingClassifier()
    gradient_boosting_classifier = GradientBoostingClassifier()
    XGB_classifier = XGBClassifier()

    # Convert y_regression to a binary classification task using a threshold
    threshold = np.mean(y)
    y_classification = (y > threshold).astype(int)

    # Split the data into training and test sets for classification
    X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X, y_classification, test_size=0.2, random_state=42)

    # Train classifiers for classification
    knn_classifier.fit(X_train_classification, y_train_classification)
    rf_classifier.fit(X_train_classification, y_train_classification)
    decision_tree_classifier.fit(X_train_classification, y_train_classification)
    gradient_boosting_classifier.fit(X_train_classification, y_train_classification)
    XGB_classifier.fit(X_train_classification, y_train_classification)

    # Evaluate classifiers using cross-validation to find their accuracies for classification
    knn_score = cross_val_score(knn_classifier, X_train_classification, y_train_classification, cv=5, scoring='accuracy').mean()
    rf_score = cross_val_score(rf_classifier, X_train_classification, y_train_classification, cv=5, scoring='accuracy').mean()
    decision_tree_score = cross_val_score(decision_tree_classifier, X_train_classification, y_train_classification, cv=5, scoring='accuracy').mean()
    gradient_boosting_score = cross_val_score(gradient_boosting_classifier, X_train_classification, y_train_classification, cv=5, scoring='accuracy').mean()
    XGB_score = cross_val_score(XGB_classifier, X_train_classification, y_train_classification, cv=5, scoring='accuracy').mean()

    # print(f"KNN Score: {knn_score}\nRandom Forest Score: {rf_score}\nGBoost Score: {gradient_boosting_score}\nDecision Tree Score: {decision_tree_score}")
    # Create a simple voting classifier for classification
    voting_clf = VotingClassifier(
        estimators=[('knn', knn_classifier), 
        ('rf', rf_classifier), 
        # ('gb', gradient_boosting_classifier), 
        ('dt', decision_tree_classifier)], 
        # ('xgb', XGB_classifier)],
        voting='soft'
    )
    voting_clf.fit(X_train_classification, y_train_classification)

    # Evaluate simple voting classifier for classification
    voting_pred = voting_clf.predict(X_test_classification)
    print(f"Voting Classifier Accuracy: {accuracy_score(y_test_classification, voting_pred)}")

    # Create a weighted voting classifier where weights are determined by classifier accuracy for classification
    weights = [knn_score, rf_score, gradient_boosting_score, decision_tree_score,XGB_score]
    weighted_voting_clf = VotingClassifier(
        estimators=[('knn', knn_classifier), ('rf', rf_classifier), ('gb', gradient_boosting_classifier), ('dt', decision_tree_classifier),('xgb', XGB_classifier)],
        voting='soft',
        weights=weights
    )
    weighted_voting_clf.fit(X_train_classification, y_train_classification)

    # Evaluate weighted voting classifier for classification
    weighted_voting_pred = weighted_voting_clf.predict(X_test_classification)
    
    # print(f"Weighted Voting Classifier Accuracy: {accuracy_score(y_test_classification, weighted_voting_pred)}")
    
    print("accuracy_score -------------------")
    voting_pred_score = accuracy_score(y_test_classification, voting_pred)
    weighted_voting_pred_score = accuracy_score(y_test_classification, weighted_voting_pred)
    
        #I obtain the accuracy of this fold
    # ac=accuracy_score(predicted,y_test)

    #I obtain the confusion matrix
    print("confusion_matrix -------------------")

    voting_TPR,voting_TNR,voting_PPV,voting_NPV,voting_FPR,voting_FNR,voting_FDR,Voting_ACC = calculate_false_positive_rate(y_test_classification, voting_pred)
    Voting_FPR = [voting_TPR,voting_TNR,voting_PPV,voting_NPV,voting_FPR,voting_FNR,voting_FDR,Voting_ACC]
    print("confusion_matrix\n", 'voting_TPR:',voting_TPR,'\nvoting_TNR:',voting_TNR,'\nvoting_PPV:',voting_PPV,'\nvoting_NPV:',voting_NPV,'\nvoting_FPR:',voting_FPR,'\nvoting_FNR:',voting_FNR,'\nvoting_FDR:',voting_FDR,'\nVoting_ACC:',Voting_ACC)

    Voting_CM  = confusion_matrix(y_test_classification, voting_pred)
    # RF_TPR,RF_TNR,RF_PPV,RF_NPV,RF_FPR,RF_FNR,RF_FDR = calculate_false_positive_rate(y_test_classification, flux_pred_random_forest)

    # KN_TPR,KN_TNR,KN_PPV,KN_NPV,KN_FPR,KN_FNR,KN_FDR = calculate_false_positive_rate(y_test_classification, flux_pred_knn)

    # DT_TPR,DT_TNR,DT_PPV,DT_NPV,DT_FPR,DT_FNR,DT_FDR = calculate_false_positive_rate(y_test_classification, flux_pred_decision_tree)

    # weighted_voting_pred_cm=confusion_matrix(y_test_classification, weighted_voting_pred)
    # Print regression metrics
    print("Regression Metrics:")
    print("Decision Tree RMSE:", mean_squared_error(y_test, flux_pred_decision_tree, squared=False))
    print("Random Forest RMSE:", mean_squared_error(y_test, flux_pred_random_forest, squared=False))
    print("KNN RMSE:", mean_squared_error(y_test, flux_pred_knn, squared=False))
    print("Gradient Boosting RMSE:", mean_squared_error(y_test, flux_pred_gradient_boosting, squared=False))
    print("XGB RMSE:", mean_squared_error(y_test, flux_pred_xgboost, squared=False))

    # Print classification metrics
    print("\nClassification Metrics:")
    print("KNN Score:", knn_score)
    print("Random Forest Score:", rf_score)
    print("Gradient Boosting Score:", gradient_boosting_score)
    print("XGB Score:", XGB_score)
    print("Decision Tree Score:", decision_tree_score)
    print("Voting Classifier Accuracy:", accuracy_score(y_test_classification, voting_pred))
    # print("Weighted Voting Classifier Accuracy:", accuracy_score(y_test_classification, weighted_voting_pred))
    
    # Print classification reports for voting classifiers

    print("\nClassification Reports for Voting Classifiers:")
    print("Voting Classifier:")
    print(classification_report(y_test_classification, voting_pred))
    # print("\nWeighted Voting Classifier:")
    # print(classification_report(y_test_classification, weighted_voting_pred))
     
    return rmse_gaussian, rmse_random_forest, rmse_knn, rmse_xgboost, rmse_decision_tree, rmse_gradient_boosting,rmse_weighted, rmse_ensemble, knn_score, rf_score,gradient_boosting_score, decision_tree_score,XGB_score,voting_pred_score,weighted_voting_pred_score, Voting_CM,Voting_FPR
    # ,TN, FP, FN, TP
    # voting_TPR,voting_TNR,voting_PPV,voting_NPV,voting_FPR,voting_FNR,voting_FDR
    # RF_TPR,RF_TNR,RF_PPV,RF_NPV,RF_FPR,RF_FNR,RF_FDR,KN_TPR,KN_TNR,KN_PPV,KN_NPV,KN_FPR,KN_FNR,KN_FDR,
    # DT_TPR,DT_TNR,DT_PPV,DT_NPV,DT_FPR,DT_FNR,DT_FDR                                            



def calculate_accuracy(y_true, y_pred):
    # Assuming y_true and y_pred are arrays of true and predicted labels respectively
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def flux_prediction(time, flux, model= None):
    # Convert time to days
#     time -= np.min(time)
    
    # Remove invalid data points
    valid_indices = np.isfinite(flux)
    valid_time = time[valid_indices]
    valid_flux = flux[valid_indices]
    
    # Additional features
    relative_time = valid_time / np.max(valid_time)  # Relative time feature
    
    # Split the data into training and test sets
    X = np.column_stack((valid_time, relative_time))
    y = valid_flux
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with scaling and regression


    # Initialize the model
    if model == 'Gaussian':
        model = GaussianProcessRegressor()
        pipeline_gaussian = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                              transformer=StandardScaler()))
        pipeline_gaussian.fit(X_train, y_train)
        flux_pred = pipeline_gaussian.predict(X_test)
    elif model == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline_random_forest = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                                    transformer=StandardScaler()))
        pipeline_random_forest.fit(X_train, y_train)
        flux_pred = pipeline_random_forest.predict(X_test)
    elif model == 'KNeighbors':
        model = KNeighborsRegressor()
        pipeline_knn = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                          transformer=StandardScaler()))
        pipeline_knn.fit(X_train, y_train)
        flux_pred = pipeline_knn.predict(X_test)
    elif model == 'XGBRegressor':
        model = XGBRegressor()
        pipeline_xgboost = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                              transformer=StandardScaler()))    
        pipeline_xgboost.fit(X_train, y_train)
        flux_pred = pipeline_xgboost.predict(X_test)

    elif model == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        pipeline_gradient_boosting = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                                        transformer=StandardScaler()))  
        pipeline_gradient_boosting.fit(X_train, y_train)
        flux_pred = pipeline_gradient_boosting.predict(X_test)
    elif model == 'DecisionTree':
        model = DecisionTreeRegressor(random_state=42)
        pipeline_decision_tree = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model,
                                                                                        transformer=StandardScaler()))   
        pipeline_decision_tree.fit(X_train, y_train)
        flux_pred = pipeline_decision_tree.predict(X_test)        


    elif model == 'MultiVote':

        model1 = GaussianProcessRegressor()
        pipeline_gaussian = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model1,
                                                                              transformer=StandardScaler()))
        pipeline_gaussian.fit(X_train, y_train)
        flux_pred1 = pipeline_gaussian.predict(X_test)

        model2 = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline_random_forest = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model2,
                                                                                    transformer=StandardScaler()))
        pipeline_random_forest.fit(X_train, y_train)
        flux_pred2 = pipeline_random_forest.predict(X_test)

        model3= KNeighborsRegressor()
        pipeline_knn = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model3,
                                                                          transformer=StandardScaler()))
        pipeline_knn.fit(X_train, y_train)
        flux_pred3 = pipeline_knn.predict(X_test)

        model4 = XGBRegressor()
        pipeline_xgboost = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model4,
                                                                              transformer=StandardScaler()))    
        pipeline_xgboost.fit(X_train, y_train)
        flux_pred4 = pipeline_xgboost.predict(X_test)

        model5 = GradientBoostingRegressor(n_estimators=100, random_state=42)
        pipeline_gradient_boosting = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model5,
                                                                                        transformer=StandardScaler()))  
        pipeline_gradient_boosting.fit(X_train, y_train)
        flux_pred5 = pipeline_gradient_boosting.predict(X_test)
        
        model6 = DecisionTreeRegressor(random_state=42)
        pipeline_decision_tree = make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=model6,
                                                                                        transformer=StandardScaler()))

        pipeline_decision_tree.fit(X_train, y_train)
        flux_pred6 = pipeline_decision_tree.predict(X_test)   

        ensemble_model = VotingRegressor([
            ('random_forest', pipeline_random_forest),
            ('knn', pipeline_knn),
            ('decision_tree', pipeline_decision_tree),
            ('gradient_boosting', pipeline_gradient_boosting),
            ('xgboost', pipeline_xgboost)
        ])
        # Impute missing values in the features using median
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        # Fit the ensemble model on the training data
        ensemble_model.fit(X_train_imputed, y_train)

        # Predict on the test data
        flux_pred = ensemble_model.predict(X_test_imputed)

        # flux_pred = (flux_pred1 + flux_pred2 + flux_pred3 + flux_pred4 ) / 4    
    else:
        print('Invalid model specified.')
        return None
    
    if model != 'MultiVote':
        time_model = X_test[:,0]
    else:
        time_model = X_test_imputed[:,0]    
    return time_model, flux_pred 
            
def radialmeasure(time, flux):
    # Convert data to Lightkurve LightCurve object
    lc = lk.LightCurve(time=time, flux=flux)

    # Calculate periodogram
    periodogram = lc.to_periodogram()
#     periodogram = lc.select_flux("flux").to_periodogram("lombscargle")
    periods = periodogram.period
    best_period = periods[periodogram.power.argmax()]

    # Calculate relative error
    relative_error = np.std(flux) / np.mean(flux)

    # Calculate radius
#     mean_radius = np.mean(radius)

    return best_period, relative_error

def plot_period_comparison(df, save_filepath, model='MM'):
    fig = plt.figure(figsize=(8, 6))
    
    P_RF = df['P_RF']
    P_GB = df['P_GB']
    P_KN = df['P_KN']
    P_DT = df['P_DT']
    P_Mu = df['P_Mu']
    Per_uperrRef = df['Per_uperrRef']
    Per_DnerrRef = df['Per_DnerrRef']
    PerExoRGB = df['PerExoRGB']
    Per_uperrExoRGP = df['Per_uperrExoRGP']
    Per_DnerrExoRGP = df['Per_DnerrExoRGP']

    P_ref = PerExoRGB  # change label for new run on 44 data sets

    # Calculate residuals
    residuals_GB = P_GB - P_ref
    residuals_RF = P_RF - P_ref
    residuals_KN = P_KN - P_ref
    residuals_DT = P_DT - P_ref
    residuals_Mu = P_Mu - P_ref

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    sorted_indices = np.argsort(P_ref)
    
    if model == 'GB':
        frame1.scatter(P_ref, P_GB, s=10, c='cyan', label='GB')
    elif model == 'RF':    
        frame1.scatter(P_ref, P_RF, s=10, c='blue', label='RF')
    elif model == 'KN':    
        frame1.scatter(P_ref, P_KN, s=10, c='purple', label='KN')
    elif model == 'DT':    
        frame1.scatter(P_ref, P_DT, s=10, c='orange', label='DT')
    elif model == 'MM':
        frame1.scatter(P_ref, P_Mu, s=30, c='red', label='Walkowicz et al. 2013', marker='o')
    else:
        frame1.scatter(P_ref, P_Mu, s=10, c='red', label='MM')

    # Add error bars
    min_val = min(min(P_ref), min(P_GB), min(P_RF), min(P_KN), min(P_DT), min(P_Mu))
    max_val = max(max(P_ref), max(P_GB), max(P_RF), max(P_KN), max(P_DT), max(P_Mu))
    frame1.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.5)

    frame1.set_ylabel('Predicted Rotation Periods (days)')
    leg2 = frame1.legend(loc='upper left')
    for t in leg2.get_texts():
        t.set_fontsize('x-large')
    frame1.set_title('Stellar Rotation Periods for Kepler Stars')
    frame1.grid(False)
    plt.tick_params(which='both', direction='in')
    plt.tick_params(labelsize=14)

    # Adding median bins data point and quartiles as shaded fill color
    bins = np.linspace(min_val, max_val, 10)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_indices = np.digitize(P_ref, bins)

    medians = []
    lower_quartiles = []
    upper_quartiles = []

    for i in range(1, len(bins)):
        bin_data = P_Mu[bin_indices == i]
        if len(bin_data) > 0:
            medians.append(np.median(bin_data))
            lower_quartiles.append(np.percentile(bin_data, 25))
            upper_quartiles.append(np.percentile(bin_data, 75))
        else:
            medians.append(np.nan)
            lower_quartiles.append(np.nan)
            upper_quartiles.append(np.nan)

    medians = np.array(medians)
    lower_quartiles = np.array(lower_quartiles)
    upper_quartiles = np.array(upper_quartiles)

    frame1.plot(bin_centers, medians, 'ko-', label='Median')
    frame1.fill_between(bin_centers, lower_quartiles, upper_quartiles, color='gray', alpha=0.5, label='1st-3rd Quartile')

    # Adding residuals plot
    frame2 = fig1.add_axes((.1, .1, .8, .2))
    frame2.plot([min_val, max_val], [0, 0], color='black', linestyle='--', alpha=0.5)

    if model == 'GB':
        frame2.scatter(P_ref[sorted_indices], residuals_GB[sorted_indices], color='cyan', s=20, label='GB', marker='o')
    elif model == 'RF':
        frame2.scatter(P_ref[sorted_indices], residuals_RF[sorted_indices], color='blue', s=20, label='RF', marker='o')
    elif model == 'KN':
        frame2.scatter(P_ref[sorted_indices], residuals_KN[sorted_indices], color='purple', s=20, label='KN', marker='o')
    elif model == 'DT':
        frame2.scatter(P_ref[sorted_indices], residuals_DT[sorted_indices], color='orange', s=20, label='DT', marker='o')
    elif model == 'MM':
        frame2.scatter(P_ref[sorted_indices], residuals_Mu[sorted_indices], color='grey', s=30, label='Walkowicz et al. 2013', marker='o')
    else:
        frame2.scatter(P_ref[sorted_indices], residuals_Mu[sorted_indices], color='red', s=20, label='MM')

    frame2.set_xlabel('Reference Rotation Periods(days)')
    frame2.set_ylabel('Residuals')
    frame2.set_ylim(-5, 5)
    frame2.grid(False)
    plt.tick_params(which='both', direction='in')
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_filepath)

# def plot_period_comparison(df, save_filepath,model='MM'):
#     fig = plt.figure(figsize=(8,6))
#     # Pl,ID,PerExoRGB,PerRef,Per_uperrExoRGP,Per_DnerrExoRGP,Per_uperrRef,
#     # Per_DnerrRef,rmse_DT,rmse_RF,rmse_KN,rmse_GB,rmse_Mu,P_RF,P_DT,P_KN,P_GB,P_Mu
#     P_RF = df['P_RF']
#     P_GB = df['P_GB']
#     P_KN = df['P_KN']
#     P_DT = df['P_DT']
#     P_Mu = df['P_Mu']
#     # P_ref = df['PerRef']
#     Per_uperrRef = df['Per_uperrRef']
#     Per_DnerrRef = df['Per_DnerrRef']
#     PerExoRGB = df['PerExoRGB']
#     Per_uperrExoRGP = df['Per_uperrExoRGP']
#     Per_DnerrExoRGP = df['Per_DnerrExoRGP']

#     n = df['Pl']
#     P_ref = PerExoRGB  # change label for new rubn on 44 data sets
#     # Calculate residuals
#     residuals_GB = P_GB - P_ref
#     residuals_RF = P_RF - P_ref
#     residuals_KN = P_KN - P_ref
#     residuals_DT = P_DT - P_ref
#     residuals_Mu = P_Mu - P_ref
#     residuals_GP = PerExoRGB - P_ref

# #     plt.figure(figsize=(8, 10))
#     fig1 = plt.figure(1)

# #     plt.subplot(2, 1, 1)
#     frame1 = fig1.add_axes((.1, .3, .8, .6))
#     sorted_indices = np.argsort(P_ref)

    
#     if model == 'GB':
#         frame1.scatter(P_ref, P_GB,  s=10, c='cyan', label='GB')
# #     P_ref, P_GB = P_ref[sorted_indices], P_GB[sorted_indices]
# #     frame1.plot(P_ref, P_GB, lw=1, color='red')
#     elif model == 'RF':    
#         frame1.scatter(P_ref, P_RF, s=10, c='blue', label='RF')
# #     P_ref, P_RF = P_ref[sorted_indices], P_RF[sorted_indices]
# #     frame1.plot(P_ref, P_RF, lw=1, color='blue')
#     elif model == 'KN':    
#         frame1.scatter(P_ref, P_KN, s=10, c='purple', label='KN')
# #     P_ref, P_KN = P_ref[sorted_indices], P_KN[sorted_indices]
# #     frame1.plot(P_ref, P_KN, lw=1, color='green')
#     elif model == 'DT':    
#         frame1.scatter(P_ref, P_DT, s=10, c='orange', label='DT')
# #     P_ref, P_DT = P_ref[sorted_indices], P_DT[sorted_indices]
# #     frame1.plot(P_ref, P_DT, lw=1, color='orange')   
#     elif model == 'MM':
#         frame1.scatter(P_ref, P_Mu, s=30, c='red', label='Walkowicz et al. 2013',marker='o')
# #     P_ref, P_Mu = P_ref[sorted_indices], P_Mu[sorted_indices]
#         # frame1.plot(P_ref, P_Mu, lw=1, color='purple')
#         # frame1.scatter(P_ref, PerExoRGB, s=10, c='green', label='GP',marker='>')
#     else:
#         frame1.scatter(P_ref, P_Mu, s=10, c='red', label='MM')
# #     P_ref, P_Mu = P_ref[sorted_indices], P_Mu[sorted_indices]
# #     frame1.plot(P_ref, P_Mu, lw=1, color='purple')    
#     # Rotate the annotation
#     # for i, txt in enumerate(n):
#     #     if ((txt == 'K78')):
#     #         frame1.annotate(txt, (P_ref[i]-0.1, P_GB[i]+P_GB[i]/4), rotation=90, verticalalignment='center',size=13)
#     #     elif  ((txt == 'K39')):
#     #         frame1.annotate(txt, (P_ref[i]-0.1, P_GB[i]+4), rotation=90, verticalalignment='center',size=13)
#     #     elif  ((txt == 'K107')):
#     #         frame1.annotate(txt, (P_ref[i]-0.1, P_GB[i]-6), rotation=90, verticalalignment='center',size=13)    
#     #     else: 
#     #         frame1.annotate(txt, (P_ref[i]-0.1, P_GB[i]-P_GB[i]/7), rotation=90, verticalalignment='center',size=13)
#         # frame1.annotate(txt, (P_ref[i]-0.1, P_Mu[i]-P_Mu[i]/5), 
#                  # rotation=90, va='center', ha='right', 
#                  # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9))
#     # Add error bars

#     # if model == 'GB':
#     #   plt.errorbar(P_ref, P_GB, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_GB']),Per_DnerrExoRGP*np.sqrt(df['rmse_GB'])], fmt='none', ecolor='cyan', alpha=0.9,capsize=4,markerfacecolor='cyan', markeredgecolor='black')
#     # elif model == 'RF':
#     #     plt.errorbar(P_ref, P_RF, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_RF']),Per_DnerrExoRGP*np.sqrt(df['rmse_RF'])], fmt='none', ecolor='blue', alpha=0.9,capsize=4,markerfacecolor='blue', markeredgecolor='black')
#     # elif model == 'KN':
#     #     plt.errorbar(P_ref, P_KN, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_KN']),Per_DnerrExoRGP*np.sqrt(df['rmse_KN'])], fmt='none', ecolor='purple', alpha=0.9,capsize=4,markerfacecolor='purple', markeredgecolor='black')
#     # elif model == 'DT':
#     #     plt.errorbar(P_ref, P_DT, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_DT']),Per_DnerrExoRGP*np.sqrt(df['rmse_DT'])], fmt='none', ecolor='orange', alpha=0.9,capsize=4,markerfacecolor='orange', markeredgecolor='black')
#     # elif model == 'MM':
#     #     # plt.errorbar(P_ref, P_Mu, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_Mu']),Per_DnerrExoRGP*np.sqrt(df['rmse_Mu'])], fmt='s', ecolor='red', alpha=0.9,capsize=4,markerfacecolor='red', markeredgecolor='black')
#     #     plt.errorbar(P_ref, P_Mu, xerr=[P_ref+Per_uperrRef,P_ref-Per_DnerrRef],yerr=[P_Mu+(Per_uperrExoRGP*np.sqrt(df['rmse_Mu'])),P_Mu-Per_DnerrExoRGP*np.sqrt(df['rmse_Mu'])], fmt='s', ecolor='red', alpha=0.9,capsize=4,markerfacecolor='red', markeredgecolor='black')

#         # plt.errorbar(P_ref, PerExoRGB, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP,Per_DnerrExoRGP], fmt='>', ecolor='green', alpha=0.9,capsize=4,markerfacecolor='green', markeredgecolor='black')

#     # else:
#     #     plt.errorbar(P_ref, P_Mu, xerr=[Per_uperrRef,Per_DnerrRef],yerr=[Per_uperrExoRGP*np.sqrt(df['rmse_Mu']),Per_DnerrExoRGP*np.sqrt(df['rmse_Mu'])], fmt='none', ecolor='purple', alpha=0.9)
#     # # Add 1:1 line
#     min_val = min(min(P_ref), min(P_GB), min(P_RF), min(P_KN), min(P_DT), min(P_Mu))
#     max_val = max(max(P_ref), max(P_GB), max(P_RF), max(P_KN), max(P_DT), max(P_Mu))
#     frame1.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.5)

# #     frame1.set_xlabel('Reference Rotation Periods(days)')
#     # frame1.set_ylabel(r'$Predicted\ Rotation\ Periods\ (days)$',size=15)
#     frame1.set_ylabel('Predicted Rotation Periods (days)', **font_style)
#     # frame1.legend()
#     leg2  =frame1.legend(loc='upper left')
#     for t in leg2.get_texts():
#         t.set_fontsize('x-large')

#     frame1.set_title('Stellar Rotation Periods for Kepler Stars', **font_style)
#     # frame1.set_ylim(0,30)
#     frame1.grid(False)
#     plt.tick_params(which='both', direction='in', **visible_ticks)
#     plt.tick_params(labelsize=14)
# #     plt.subplot(2, 1, 2)
#     frame2 = fig1.add_axes((.1, .1, .8, .2))

#     frame2.plot([min_val, max_val], [0, 0], color='black', linestyle='--', alpha=0.5)

#     if model == 'GB':
#         frame2.scatter(P_ref[sorted_indices], residuals_GB[sorted_indices],color='cyan', s=20, label='GB',marker='o')
#     elif model == 'RF':
#         frame2.scatter(P_ref[sorted_indices], residuals_RF[sorted_indices],color='blue', s=20, label='RF',marker='o')
#     elif model == 'KN':
#         frame2.scatter(P_ref[sorted_indices], residuals_KN[sorted_indices],color='purple', s=20, label='KN',marker='o')
#     elif model == 'DT':
#         frame2.scatter(P_ref[sorted_indices], residuals_DT[sorted_indices],color='orange', s=20, label='DT',marker='o')
#     elif model == 'MM':
#         frame2.scatter(P_ref[sorted_indices], residuals_Mu[sorted_indices], color = 'grey', s=30, label='Walkowicz et al. 2013',marker='o')
#         # frame2.plot(P_ref[sorted_indices], residuals_Mu[sorted_indices], color = 'red',lw = 1)
#         # frame2.scatter(P_ref[sorted_indices], residuals_GP[sorted_indices], color = 'green', s=20, label='GP',marker='>')
#     else:
#         frame2.scatter(P_ref[sorted_indices], residuals_Mu[sorted_indices], color='red', s=20, label='MM')
#     frame2.set_xlabel('Reference Rotation Periods(days)', **font_style)
#     frame2.set_ylabel('Residuals', **font_style)
#     frame2.set_ylim(-5,5)
# #     frame2.legend()
# #     frame2.title('Residual Plot')
#     frame2.grid(False)
#     plt.tick_params(which='both', direction='in', **visible_ticks)
#     plt.tick_params(labelsize=14)
#     plt.tight_layout()

#     # Save the plot
#     plt.savefig(save_filepath)
#     #plt.show()

def ls_periodogram(x, y, sector=None, min_period=1, max_period=100.0, xlim_min=None, xlim_max=None, bls=None,target_id=None,Model=None):
    results = lomb_scargle_estimator(x, y, max_peaks=20, min_period=min_period, max_period=max_period, samples_per_peak=50)

    # Select peak, which represents rotation period estimate
    peak = results["peaks"][0]
    freq, power = results["periodogram"]

    # Plot LS periodogram
    plt.figure(figsize=(10, 5))
    plt.annotate(f"LS period = {peak['period']:.4f} d\nperiod uncertainty= {peak['period_uncert']:.4f}", (0.98, 0.8), xycoords="axes fraction", xytext=(5, -5), textcoords="offset points", va="top", ha="right", fontsize=12)
    plt.plot(1 / freq, power, "k")

    # Fit a Gaussian to the periodogram peak
    from scipy.optimize import curve_fit

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, 1 / freq, power, p0=[power.max(), peak["period"], 0.1])
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
    fwhm_err = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(pcov[2,2])

    # Overplot the Gaussian fit and the FWHM error
    plt.plot(1 / freq, gaussian(1 / freq, *popt), 'r--', label=f'Gaussian fit, FWHM={fwhm:.3f} ± {fwhm_err:.3f} days')
    plt.fill_between(1 / freq, gaussian(1 / freq, *popt) - fwhm_err, gaussian(1 / freq, *popt) + fwhm_err, color='r', alpha=0.2)

#     if bls:
#         plt.axvline(bls, color="r", lw=4, alpha=0.3, label="BLS Period")
    plt.axvline(peak["period"], color="k", lw=4, alpha=0.3, label=f"{Model}-LS Period ({target_id})")
    if xlim_min is not None and xlim_max is not None:
        plt.xlim(xlim_min, xlim_max)
    else:
        plt.xlim((1 / freq).min(), (1 / freq).max())
    plt.yticks([])
    plt.xlabel("period [days]",size = 20)
    plt.ylabel("power",size = 20)
#     if sector:
#         plt.title(f"{pl_hostname} ({target_id}): LS Periodogram for sector={sector}, corr={corr}")
#     else:
#         plt.title(f"{pl_hostname} ({target_id}): LS Periodogram for sector=all, corr={corr}")
    leg = plt.legend(loc="upper right")
    for t in leg.get_texts():
        t.set_fontsize('x-large')

    # Save plot
    if target_id !=None:
        ls_plot = f'./out_data/ls_period_{target_id}.png'
    plt.savefig(ls_plot)
#     ls_plot.close()

    return peak["period"],peak['period_uncert']
# Victoria
# ---------------------------------------------------------------------------------------
# Download all data for a star
def download_all(target_id, mission):

    if mission == "TESS":
        tpf_2mins = lk.search_targetpixelfile(target_id, mission="TESS").download_all()
        lcfs = lk.search_lightcurvefile(target_id, mission="TESS").download_all()

    elif mission == "Kepler":
        tpf_2mins = []
        lcfs = lk.search_lightcurvefile(target_id, mission="Kepler").download_all()

    print(lcfs)

    return tpf_2mins, lcfs

# Download selected data for a star
def download_selected(target_id, mission, selected):

    tpf_2mins = []
    lcfs = []

    for s in selected:
        if mission == "TESS":
            tpf_2min = lk.search_targetpixelfile(target_id, mission="TESS", sector=s).download()
            lcf = lk.search_lightcurvefile(target_id, mission="TESS", sector=s).download()
            tpf_2mins.append(tpf_2min)
            lcfs.append(lcf)
        elif mission == "Kepler":
            tpf_2min = lk.search_targetpixelfile(target_id, mission="Kepler", quarter=s).download()
            lcf = lk.search_lightcurvefile(target_id, mission="Kepler", quarter=s).download()
            tpf_2mins.append(tpf_2min)
            lcfs.append(lcf)

    return tpf_2mins, lcfs

# Remove missing values from target pixel file
def remove_all_nans(self, tpf):
    
    test = np.isnan(self.flux_err)
    self = self[~np.isnan(self.flux)]
    self = self[~np.isnan(self.flux_err)]
    self = self[~np.isnan(self.time)]
    
    tpf = tpf[~test]

    return self, tpf

# Apply Regression Correction to lightcurve
def make_reg_corrections(tpf_2min):

    # Extract lightcurve from target pixel file
    aper = tpf_2min.pipeline_mask
    raw_lc = tpf_2min.to_lightcurve()
    raw_lc, tpf_2min = remove_all_nans(raw_lc, tpf_2min)

    # Make a design matrix
    dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(5).append_constant()

    # Apply Regression Correction
    reg = lk.RegressionCorrector(raw_lc)
    lc_reg = reg.correct(dm)

    return lc_reg

# Apply PLD Correction to lightcurve
def make_pld_corrections(tpf_2min):

    # Obtain lightcurve from target pixel file
    aper = tpf_2min.pipeline_mask
    raw_lc = tpf_2min.to_lightcurve()
    raw_lc, tpf_2min = remove_all_nans(raw_lc, tpf_2min)
    
    # Apply PLD Correction
    pld = lk.TessPLDCorrector(tpf_2min)
    lc_pld = pld.correct()

    return lc_pld

# Stitch sectors of data together for a star
# Apply preprocessing (binning, correcting, normalizing)
def stitch_lightcurves(lcfs, tpf_2mins, bins=None, corr="sap"):

    time = []
    flux = []
    flux_err = []

    for i,lcf in enumerate(lcfs):

      # Apply corrections
      if corr == "sap":
        corrected = lcf.SAP_FLUX.remove_nans()
      elif corr == "pdcsap":
        corrected = lcf.PDCSAP_FLUX.remove_nans()
      elif corr == "reg":
        corrected = make_reg_corrections(tpf_2mins[i])
      elif corr == "pld":
        corrected = make_pld_corrections(tpf_2mins[i])

      # Bin lightcurve
      if bins:
          corrected = corrected.bin(binsize=bins)

      # Normalize lightcurve
      time = np.concatenate((time,corrected.time))
      norm_flux = (corrected.flux / np.median(corrected.flux) - 1) * 1e3
      flux = np.concatenate((flux,norm_flux))
      norm_flux_err = corrected.flux_err * 1e3 / np.mean(corrected.flux)
      flux_err = np.concatenate((flux_err,norm_flux_err))

      if i == 0:
        lc = corrected.normalize()
      else:
        lc = lc.append(corrected.normalize())

    return lc, time, flux, flux_err


def get_correct_depth(tls_period, time, flux):

  # Create constrained period grid based on tls_period
  period_grid = np.exp(np.linspace(np.log(tls_period-0.1), np.log(tls_period+0.1), 5000))

  # Apply BLS
  bls = BoxLeastSquares(time, flux)
  bls_power = bls.power(period_grid, 0.01, oversample=20)

  # Save the highest peak as the planet candidate
  index = np.argmax(bls_power.power)

  bls_period = bls_power.period[index]
  bls_t0 = bls_power.transit_time[index]
  bls_depth = bls_power.depth[index]

  print(bls_period, bls_t0, bls_depth)

  return bls_period, bls_t0, bls_depth

# Apply Box Least Squares algorithm to lightcurve to get transit parameters
def BLS(flattened_lc, norm_time, norm_flux, sectors=None):

  # Range of periods to search for
  period_grid = np.exp(np.linspace(np.log(0.1), np.log(50), 50000))

  # Apply BLS to flattened lc
  bls = BoxLeastSquares(flattened_lc.time, flattened_lc.flux)
  bls_power = bls.power(period_grid, 0.01, oversample=20)

  # Save the highest peak as the planet candidate
  index = np.argmax(bls_power.power)

  # Get transit period and time from flattened lc
  bls_period = bls_power.period[index]
  bls_t0 = bls_power.transit_time[index]

  # Apply BLS to normalized lc
  norm_bls = BoxLeastSquares(norm_time, norm_flux)
  norm_bls_power = norm_bls.power(period_grid, 0.01, oversample=20)

  # Get transit depth from normalized lc
  bls_depth = norm_bls_power.depth[index]

  # # To store plot in Document
  # # Document.add_heading("Box Least Squares")
  # bls_plot = BytesIO()

  # fig, axes = plt.subplots(2, 1, figsize=(10, 10))

  # # Plot the periodogram
  # ax = axes[0]
  # ax.axvline(np.log10(bls_period), color="C1", lw=5, alpha=0.8)
  # ax.plot(np.log10(bls_power.period), bls_power.power, "k")
  # ax.annotate(
  #     "bls period = {0:.4f} d".format(bls_period)+"\nbls t0 = {0:.4f}".format(bls_t0)+"\nbls depth = {0:.4f}".format(bls_depth),
  #     (0, 1),
  #     xycoords="axes fraction",
  #     xytext=(5, -5),
  #     textcoords="offset points",
  #     va="top",
  #     ha="left",
  #     fontsize=12,
  # )
  # ax.set_ylabel("bls power")
  # ax.set_yticks([])
  # ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
  # ax.set_xlabel("log10(period)")

  # if sectors:
  #   ax.set_title(pl_hostname+" ("+target_id+"): BLS for sectors="+str(sectors)+", corr="+corr)
  # else:
  #   ax.set_title(pl_hostname+" ("+target_id+"): BLS for sectors=all, corr="+corr)

  # # Plot the folded transit
  # ax = axes[1]
  # x_fold = (time - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
  # m = np.abs(x_fold) < 0.4
  # ax.plot(x_fold[m], flux[m], ".k")

  # # Overplot the phase binned light curve
  # bins = np.linspace(-0.41, 0.41, 32)
  # denom, _ = np.histogram(x_fold, bins)
  # num, _ = np.histogram(x_fold, bins, weights=flux)
  # denom[num == 0] = 1.0
  # ax.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1")

  # ax.set_ylabel("de-trended flux [ppt]")
  # ax.set_xlabel("time since transit");

  # # Add plot to Document
  # plt.savefig(bls_plot)
  # # Document.add_picture(bls_plot,width=Inches(6))
  # bls_plot.close()

  return bls_period, bls_t0, bls_depth

# Apply Transit Least Squares algorithm to lightcurve to get transit parameters
def TLS(flattened_lc, time, flux, id_num, sectors=None):

  # Obtain stellar parameters based on target id
  if mission=="TESS":
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=id_num)
  elif mission=="Kepler":
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(KIC_ID=id_num)
  print('Searching with limb-darkening estimates using quadratic LD (a,b)=', ab)
  print(ab, mass, mass_min, mass_max, radius, radius_min, radius_max)

  # Apply Transit Least Squares to flattened lightcurve
  tls_model = transitleastsquares(time, flattened_lc.flux)
  tls_results = tls_model.power(u=ab, period_min=0.1, period_max=50)
  print("tls period:", tls_results.period, "period uncertainty:", tls_results.period_uncertainty, "T0:", tls_results.T0)

  # Apply Box Least Squares to normal lightcurve using period found by TLS
  bls_period, bls_t0, bls_depth = get_correct_depth(tls_results.period, time, flux)

  # For storing plot in Document
  # Document.add_heading("Transit Least Squares")
  tls_plot = BytesIO()

  # Plot TLS results
  plt.figure(figsize=(10,5))
  ax = plt.gca()
  ax.axvline(tls_results.period, alpha=0.4, lw=3)
  plt.xlim(np.min(tls_results.periods), np.max(tls_results.periods))
  for n in range(2, 10):
      ax.axvline(n*tls_results.period, alpha=0.4, lw=1, linestyle="dashed")
      ax.axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed")
  ax.annotate(
      "tls period = {0:.4f} d".format(tls_results.period)+"\ntls t0 = {0:.4f}".format(tls_results.T0)+"\nbls period = {0:.4f} d".format(bls_period)+"\nbls t0 = {0:.4f}".format(bls_t0)+"\nbls depth = {0:.4f}".format(bls_depth),
      (0.98, 1),
      xycoords="axes fraction",
      xytext=(5, -5),
      textcoords="offset points",
      va="top",
      ha="right",
      fontsize=12,
  )
  plt.ylabel(r'SDE')
  plt.xlabel('Period (days)')
  plt.plot(tls_results.periods, tls_results.power, color='black', lw=0.5)

  if sectors:
    plt.title(pl_hostname+" ("+target_id+"): TLS for sectors="+str(sectors)+", corr="+corr)
  else:
    plt.title(pl_hostname+" ("+target_id+"): TLS for sectors=all, corr="+corr)

  # Add plot to Document
  plt.savefig(tls_plot)
  # Document.add_picture(tls_plot,width=Inches(7))
  tls_plot.close()

  return bls_period, bls_t0, bls_depth, tls_results

# Apply LS Periodogram to lightcurve
# def ls_periodogram(x, y, sector=None, min_period=1, max_period=100.0, xlim_min=None, xlim_max=None, bls=None):

#   results = lomb_scargle_estimator(
#       x, y, max_peaks=20, min_period=min_period, max_period=max_period, samples_per_peak=50
#   )

#   # Select peak, which represents rotation period estimate
#   peak = results["peaks"][0]
#   freq, power = results["periodogram"]

#   # For storing plot in Document
#   # Document.add_heading("LS Periodogram")
#   # ls_plot = BytesIO()

#   # # Plot LS periodogram
#   # plt.figure(figsize=(10,5))
#   # plt.annotate(
#   #     "LS period = {0:.4f} d".format(peak["period"])+"\nperiod uncertainty= {0:.4f}".format(peak['period_uncert']),
#   #     (0.98, 0.8),
#   #     xycoords="axes fraction",
#   #     xytext=(5, -5),
#   #     textcoords="offset points",
#   #     va="top",
#   #     ha="right",
#   #     fontsize=12,
#   # )
#   # plt.plot(1 / freq, power, "k")
#   # if bls:
#   #   plt.axvline(bls, color="r", lw=4, alpha=0.3, label="BLS Period")
#   # plt.axvline(peak["period"], color="k", lw=4, alpha=0.3, label="LS Period")
#   # if xlim_min != None and xlim_max != None:
#   #   plt.xlim(xlim_min,xlim_max)
#   # else:
#   #   plt.xlim((1 / freq).min(), (1 / freq).max())
#   # plt.yticks([])
#   # plt.xlabel("period [days]")
#   # plt.ylabel("power");
#   # if sector:
#   #   plt.title(pl_hostname+" ("+target_id+"): LS Periodogram for sector="+sector+", corr="+corr)
#   # else:
#   #   plt.title(pl_hostname+" ("+target_id+"): LS Periodogram for sector=all, corr="+corr)
#   # plt.legend(loc="upper right")

#   # # Add plot to Document
#   # plt.savefig(ls_plot)
#   # # Document.add_picture(ls_plot,width=Inches(7))
#   # ls_plot.close()

#   return peak["period"]

# Apply Autocorrelation function to lightcurve
def ACF(time, flux, yerr, sectors=None):

    acorr = autocorr_estimator(time,flux,yerr=yerr,min_period=1,max_period=100,oversample=5.0,smooth=2.0,max_peaks=10)

    # Select peak, which represents rotation period estimate
    acf_period = np.nan
    if acorr['peaks']:
      acf_period = acorr['peaks'][0]['period']
    
    print("rot. period:",acf_period)
    
    delta_t = np.median(np.diff(time))

    # One dimensional linear interpolation
    y_interp = np.interp(np.arange(time.min(), time.max(), delta_t), time, flux)
    emp_acorr = emcee.autocorr.function_1d(y_interp) * np.var(y_interp)
    tau = np.arange(len(emp_acorr)) * delta_t
    
    # For storing plot in Document
    # Document.add_heading("Autocorrelation")
    acf_plot = BytesIO()

    # Plot ACF
    plt.figure(figsize=(10,5))
    plt.annotate(
      "ACF period = {0:.4f} d".format(acf_period),
      (0.98, 0.8),
      xycoords="axes fraction",
      xytext=(5, -5),
      textcoords="offset points",
      va="top",
      ha="right",
      fontsize=12,
    )
    plt.plot(tau, emp_acorr)
    plt.axvline(acf_period, color="k", alpha=0.5, label="ACF Period")
    plt.axvline(2*acf_period, color="k", alpha=0.5)
    plt.axvline(3*acf_period, color="k", alpha=0.5)
    plt.ylabel(r"$\left< k(\tau) \right>$")
    plt.xlabel(r"$\tau$")
    plt.title(pl_hostname+" ("+target_id+"): ACF")
    if sectors:
        plt.title(pl_hostname+" ("+target_id+"): ACF for sectors="+str(sectors)+", corr="+corr)
    else:
        plt.title(pl_hostname+" ("+target_id+"): ACF for sectors=all, corr="+corr)
    plt.legend()

    # Add plot to Document
    plt.savefig(acf_plot)
    # Document.add_picture(acf_plot,width=Inches(7))
    acf_plot.close()

    return acf_period

def mask_transits(time, flux, tls_results):

    intransit = transit_mask(time, tls_results.period, 2 * tls_results.duration, tls_results.T0)
    flux_ootr = flux[~intransit]

    # Document.add_heading("Masked Lightcurve")
    masked_lightcurve_plot = BytesIO()

    plt.figure(figsize=(20,5))
    plt.plot(time[~intransit],flux[~intransit])

    # Add plot to Document
    plt.savefig(masked_lightcurve_plot)
    # Document.add_picture(masked_lightcurve_plot, width=Inches(7))
    masked_lightcurve_plot.close()

    time = time[~intransit]
    flux = flux[~intransit]

    return time, flux

# Plot components of GP model
def plot_GP_model(soln, mask=None, transit=True, rot=True):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    if rot and not transit:
      fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

      # Plot stellar activtiy
      ax = axes[0]
      ax.plot(x[mask], y[mask], "k", label="data")
      gp_mod = soln["gp_pred"] + soln["mean"]
      ax.plot(x[mask], gp_mod, color="C2", label="gp model", linewidth=7.0, alpha=0.5)
      ax.legend(fontsize=10)
      ax.set_ylabel("relative flux")

      # Plot residuals
      ax = axes[1]
      ax.plot(x[mask], y[mask] - gp_mod, "k")
      ax.axhline(0, color="#aaaaaa", lw=1)
      ax.set_ylabel("residuals [ppt]")
      ax.set_xlim(x[mask].min(), x[mask].max())
      ax.set_xlabel("time [days]")

    else:
      fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

      # Plot stellar activity
      ax = axes[0]
      ax.plot(x[mask], y[mask], "k", label="data")
      gp_mod = soln["gp_pred"] + soln["mean"]
      ax.plot(x[mask], gp_mod, color="C2", label="gp model", linewidth=7.0, alpha=0.5)
      ax.legend(fontsize=10)
      ax.set_ylabel("relative flux")

      # Plot exoplanet transits
      ax = axes[1]
      ax.plot(x[mask], y[mask] - gp_mod, "k", label="de-trended data")
      for i, l in enumerate("b"):
          mod = soln["light_curves"][:, i]
          ax.plot(x[mask], mod, label="planet {0}".format(l), linewidth=6.0, alpha=0.5)
      ax.legend(fontsize=10, loc=3)
      ax.set_ylabel("de-trended flux [ppt]")

      # Plot residuals
      ax = axes[2]
      mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
      ax.plot(x[mask], y[mask] - mod, "k")
      ax.axhline(0, color="#aaaaaa", lw=1)
      ax.set_ylabel("residuals [ppt]")
      ax.set_xlim(x[mask].min(), x[mask].max())
      ax.set_xlabel("time [days]")

    return fig

# Display model parameters in table
def display_model_params(model_params, var_names):

    table_data = {}

    for name in var_names:
      if name.startswith("log"):
        new_name = name[3:]
        table_data[new_name] = np.exp(model_params[name])
      else:
        table_data[name] = model_params[name].item()
        if name == "r_pl":
          table_data["r_pl_jup"] = (model_params[name].item() * u.R_sun).to(u.R_jup).value

    params_table = pd.DataFrame(data=table_data, index=[0])
    display(params_table)

    # Add GP parameter values to Document
    # Document.add_heading("MAP Parameters")
    lines = ""
    for c in params_table.columns:
      lines += c+": "+str(params_table[c].iloc[0].round(4))+"\n"
    # para = Document.add_paragraph(lines)

def clip_outliers(x, y, map_params0, rot, transit):

  if transit:
    mod = (
        map_params0["gp_pred"]
        + map_params0["mean"]
        + np.sum(map_params0["light_curves"], axis=-1)
    )
  else:
    if rot:
      mod = (
          map_params0["gp_pred"]
          + map_params0["mean"]
      )

  resid = y - mod
  rms = np.sqrt(np.median(resid ** 2))
  mask = np.abs(resid) < 5 * rms

  # Document.add_heading("Outliers")

  outliers_plot = BytesIO()

  plt.figure(figsize=(10, 5))
  plt.plot(x, resid, "k", label="data")
  plt.plot(x[~mask], resid[~mask], "xr", label="outliers")
  plt.axhline(0, color="#aaaaaa", lw=1)
  plt.ylabel("residuals [ppt]")
  plt.xlabel("time [days]")
  plt.legend(fontsize=12, loc=3)
  plt.xlim(x.min(), x.max());

  plt.savefig(outliers_plot)
  # Document.add_picture(outliers_plot,width=Inches(7))
  outliers_plot.close()

  return mask

# Get mean and quantiles from samples
def get_statistics(samples, transit=True, rot=True):
  
    param_names = []
    statistics = {}

    for c in samples.columns:
      
      # Calculate mean and quantiles for parameter
      mean = samples[c].mean()
      std = samples[c].std()
      q = np.percentile(samples[c], [16, 50, 84])
      lb = np.diff(q)[0]
      ub = np.diff(q)[1]

      # If sampled from log space, convert param name and value
      param = c
      if param.startswith("log"):
        mean = np.exp(mean)
        std = np.exp(std)
        lb = np.exp(lb)
        ub = np.exp(ub)
        param = c[3:]
      
      # Store parameter names
      if param == 'r_pl':
        param_names.append('r_pl_jup')
      else:
        param_names.append(param)

      # Store statistics
      statistics[param] = round(mean,5)
      statistics[param+" std"] = round(std,5)
      statistics[param+" lb"] = round(lb,5)
      statistics[param+" ub"] = round(ub,5)

    # Add parameter for Jupiter radius
    if transit:
      jup_rad_mean = (statistics['r_pl'] * u.R_sun).to(u.R_jup).value
      jup_rad_std = (statistics['r_pl std'] * u.R_sun).to(u.R_jup).value
      jup_rad_lb = (statistics['r_pl lb'] * u.R_sun).to(u.R_jup).value
      jup_rad_ub = (statistics['r_pl ub'] * u.R_sun).to(u.R_jup).value

      statistics['r_pl_jup'] = round(jup_rad_mean,5)
      statistics['r_pl_jup std'] = round(jup_rad_std,5)
      statistics['r_pl_jup lb'] = round(jup_rad_lb,5)
      statistics['r_pl_jup ub'] = round(jup_rad_ub,5)

    return statistics, param_names

# Display results along side reference values in a table
def results_table(gp_results, ref_results, param_names):

  data = []

  for param in param_names:

    # Get GP value and interval
    gp_value = gp_results[param]
    gp_error = (abs(gp_results[param+" lb"]), abs(gp_results[param+" ub"]))

    # Initialize ref value, error, and name
    ref_value = ""
    ref_error = ""
    ref_name = ""

    # Check if param is in ref table
    if param in ref_results.columns:

      # Set ref value, if it exists
      if np.isfinite(float(ref_results[param].iloc[0])):
        ref_value = float(ref_results[param].iloc[0])

        # Set ref error, if it exists
        if np.isfinite(ref_results[param+" lb"].iloc[0]):
          ref_error = (abs(ref_results[param+" lb"].iloc[0].round(4)), +abs(ref_results[param+" ub"].iloc[0].round(4)))

        # Set ref name
        if param == "rotperiod":
          ref_name = ref_results['prot_refname'].iloc[0]
        elif param in ["m_star","r_star"]:
          ref_name = ref_results['st_refname'].iloc[0]
        else:
          ref_name = ref_results['pl_refname'].iloc[0]

    data.append([param, gp_value, gp_error, ref_value, ref_error, ref_name])

  results = pd.DataFrame(data, columns = ['param','gp_mean','gp_error','ref_mean','ref_error','ref_name'])

  return results

# Plot error bars for parameter
def print_error_bars(results, param, sectors):

  # Get GP results for parameter
  x = [0]
  y = [float(results[results['param']==param]['gp_mean'].iloc[0])]
  lb = [float(results[results['param']==param]['gp_error'].iloc[0][0])]
  ub = [float(results[results['param']==param]['gp_error'].iloc[0][1])]

  # For storing plot in Document
  error_bars_plot = BytesIO()

  # Plot GP results for parameter
  plt.figure(figsize=(7,7))
  plt.errorbar(x, y, [lb,ub], capsize=5, marker='o', ls="None", label="GP model")

  # Get Reference information for parameter
  ref_value = results[results['param']==param]['ref_mean'].iloc[0]
  ref_error = results[results['param']==param]['ref_error'].iloc[0]
  ref_name = results[results['param']==param]['ref_name'].iloc[0]

  # Plot Reference information for parameter
  if ref_value:
      plt.hlines(float(ref_value), min(x)-1, max(x)+1, alpha=0.15, label=ref_name)
      if ref_error:
          plt.fill_between([min(x)-1,max(x)+1], float(ref_value) - float(ref_error[0]), float(ref_value) + float(ref_error[1]), alpha=0.15)
  plt.title(pl_hostname+": MCMC guess "+param, fontsize=25)
  plt.ylabel(param, fontsize=20)
  plt.xlabel("Sectors", fontsize=20)
  plt.xticks(x,sectors)
  plt.legend()

  # Add plot to Document
  plt.savefig(error_bars_plot)
  # Document.add_picture(error_bars_plot,width=Inches(2))
  error_bars_plot.close()

def get_model_components(trace, mask=None, transit=True, rot=True):
  
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    # Document.add_heading("Model Components")

    # For storing plot in Document
    model_components_plot = BytesIO()

    if transit:
      fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

      ax = axes[0]
      ax.plot(x[mask], y[mask], "k", label="data")

      gp_mod = np.median(trace["gp_pred"] + trace["mean"][:, None], axis=0)
      ax.plot(x[mask], gp_mod, color="C2", label="gp model", linewidth=7.0, alpha=0.5)
      ax.legend(fontsize=10)
      ax.set_ylabel("relative flux")

      lc = np.median(np.sum(trace["light_curves"], axis=-1), axis=0)
      ax = axes[1]
      ax.plot(x[mask], y[mask] - gp_mod, "k", label="de-trended data")
      ax.plot(x[mask], lc, label="planet", linewidth=7.0, alpha=0.5)
      ax.legend(fontsize=10, loc=3)
      ax.set_ylabel("de-trended flux [ppt]")

      ax = axes[2]
      mod = gp_mod + lc
      ax.plot(x[mask], y[mask] - mod, "k")
      ax.axhline(0, color="#aaaaaa", lw=1)
      ax.set_ylabel("residuals [ppt]")
      ax.set_xlim(x[mask].min(), x[mask].max())
      ax.set_xlabel("time [days]")

      model_components = {
        'x': x[mask],
        'y': y[mask],
        'gp_mod': gp_mod,
        'light_curves': lc,
        'residuals': y[mask] - gp_mod - lc
      }
    else:
      if rot:
        fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

        ax = axes[0]
        ax.plot(x[mask], y[mask], "k", label="data")

        gp_mod = np.median(trace["gp_pred"] + trace["mean"][:, None], axis=0)
        ax.plot(x[mask], gp_mod, color="C2", label="gp model", linewidth=7.0, alpha=0.5)
        ax.legend(fontsize=10)
        ax.set_ylabel("relative flux")

        ax = axes[1]
        ax.plot(x[mask], y[mask] - gp_mod, "k")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("residuals [ppt]")
        ax.set_xlim(x[mask].min(), x[mask].max())
        ax.set_xlabel("time [days]")

        model_components = {
          'x': x[mask],
          'y': y[mask],
          'gp_mod': gp_mod,
          'residuals': y[mask] - gp_mod
        }

    # Add plot to Document
    plt.savefig(model_components_plot)
    # Document.add_picture(model_components_plot,width=Inches(7))
    model_components_plot.close()

    return model_components
