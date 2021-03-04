
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold



# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    #parser.add_argument('--n-estimators', type=int, default=10)
    #parser.add_argument('--min-samples-leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='df_train.csv')
    parser.add_argument('--test-file', type=str, default='df_test.csv')
    #parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    #parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target
    args, _ = parser.parse_known_args()
    print(args.train)
    SEED = 42
    print('reading data')
    df_train = pd.read_csv(os.path.join(args.train, args.train_file))
    df_test = pd.read_csv(os.path.join(args.test, args.test_file))
    missing_cols = list(set(df_train.columns) - set(df_test.columns))
    for col in missing_cols:
        df_test[col] = 0
#     df_train = pd.read_csv(args.train)
#     df_test = pd.read_csv(args.test)
    drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']
    
    X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train['Survived'].values
    X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))

    print('X_train shape: {}'.format(X_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    print('building training and testing datasets')
    leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=SEED,
                                           n_jobs=-1,
                                           verbose=1)
    N = 5
    oob = 0
    probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
    importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=df_train.drop(drop_cols, axis=1).columns)
    fprs, tprs, scores = [], [], []

    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print('Fold {}\n'.format(fold))

        # Fitting the model
        leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])

        # Computing Train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], leaderboard_model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], leaderboard_model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)  

        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)

        # X_test probabilities
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]
        importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_

        oob += leaderboard_model.oob_score_ / N
        print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))   
    
    print('Average OOB Score: {}'.format(oob))

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(leaderboard_model, path)
    print('model persisted at ' + path)
