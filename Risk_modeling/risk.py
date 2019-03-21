import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier

import gc

N_SPLIT   = 5

def build_input():
    input_directory = "~/Dropbox/CS/Kaggle/Home_credit_default/input/"
    
    print("[0] Getting the application data..")
    train = pd.read_csv(input_directory + "application_train.csv")

    # Columns to drop
    avg_columns = [c for c in train.columns if "AVG" in c]
    mode_columns = [c for c in train.columns if "MODE" in c]
    medi_columns = [c for c in train.columns if "MEDI" in c]

    drop_columns = avg_columns + mode_columns
    data = train.drop(drop_columns, axis=1) 
    
    del train
    gc.collect()

    buro = get_buro(input_directory)
    prev = get_prev(input_directory)
    inst = get_inst(input_directory)

    print("Combining the buro, previous applications, and installment payment data..")
    data = data.merge(right=buro, how='left', on='SK_ID_CURR')
    data = data.merge(right=inst, how='left', on='SK_ID_CURR')
    data = data.merge(right=prev.reset_index(), how='left', on='SK_ID_CURR')
    
    X = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y = data['TARGET']

    X = pd.get_dummies(X)
    # X.fillna(X.median(), inplace=True)

    del data
    gc.collect()

    return train_test_split(X, y, test_size=0.1)

def get_buro(input_directory):

    print("[1] Getting the bureau data..")
    data = pd.read_csv(input_directory + "bureau.csv")
    data.drop(['DAYS_CREDIT_UPDATE', 
               'AMT_ANNUITY',
               'AMT_CREDIT_SUM',
               'AMT_CREDIT_SUM_DEBT',
               'AMT_CREDIT_SUM_LIMIT',
               'DAYS_CREDIT_ENDDATE',
               'DAYS_ENDDATE_FACT'
               ], axis=1, inplace=True)
    buro = pd.get_dummies(data)

    numeric_feats = [col for col in data.columns if data[col].dtype != 'object']
    numeric_feats.remove('SK_ID_CURR')
    numeric_feats.remove('SK_ID_BUREAU')

    # We are concerned with select categorical variables
    categoric_feats = ['CREDIT_ACTIVE_Active']
    credit_type_cat = list(data.CREDIT_TYPE.value_counts()[:5].index)
    credit_type_cat = ['CREDIT_TYPE_' + x for x in credit_type_cat]
    categoric_feats += credit_type_cat

    del data
    gc.collect()

    gb = buro.groupby('SK_ID_CURR')
    
    avgs = gb.mean()
    maxs = gb.max()
    sums = gb.sum()

    df = pd.DataFrame({'SK_ID_CURR':avgs.index})
    
    for col in numeric_feats:
        df['max_' + col] = maxs[col]
        df['mean_' + col] = avgs[col]

    for col in categoric_feats:
        df['mean_' + col] = avgs[col]
        df['sum_' + col]  = sums[col]

    df.columns = ['SK_ID_CURR'] + ['buro_' + f_ for f_ in df.columns if f_ != 'SK_ID_CURR']

    del buro
    gc.collect()

    return df

def get_prev(input_directory):

    print("[2] Getting the previous application data..")
    data = pd.read_csv(input_directory + "previous_application.csv")
    data.drop(["RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED", "NAME_TYPE_SUITE", 
               #"PRODUCT_COMBINATION", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", 
               "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_DECISION",
               "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION", 
               "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"], axis=1, inplace=True)

    prev = pd.get_dummies(data)
    del data
    gc.collect()

    gb = prev.groupby("SK_ID_CURR")

    avg = gb.mean()
    avg.drop('SK_ID_PREV', axis=1, inplace=True)
    avg.columns = ['prev_' + f_ for f_ in avg.columns]

    del prev
    gc.collect()

    return avg

def get_inst(input_directory):

    print('[3] Getting installment payment data..')
    inst = pd.read_csv(input_directory + 'installments_payments.csv')
    inst['payday_diff'] = inst.DAYS_INSTALMENT - inst.DAYS_ENTRY_PAYMENT

    # Count payments
    counts = inst.SK_ID_CURR.value_counts()
    counts = pd.DataFrame({'SK_ID_CURR':counts.index, 'NUM_PAYMENTS':counts.values})

    # Count late payments
    late_counts = inst[inst.payday_diff < 0].SK_ID_CURR.value_counts()
    late_counts = pd.DataFrame({'SK_ID_CURR': late_counts.index, 'NUM_LATE_PAYMENT':late_counts.values})

    # Calculate fraction of late payments
    agg_inst = counts.merge(late_counts, how='left', on='SK_ID_CURR')
    agg_inst['FRAC_LATE_PAY'] = agg_inst.NUM_LATE_PAYMENT/agg_inst.NUM_PAYMENTS
    agg_inst.columns = ['inst_' + f_ if f_ is not 'SK_ID_CURR' else f_ for f_ in agg_inst.columns]

    del inst, counts, late_counts
    gc.collect()

    return agg_inst

def train_model(X_train, X_test, y_train, y_test, folds_idx):

    print("Training LGBM classifier..")
    y_pred_train = np.zeros(X_train.shape[0])
    y_pred_test  = np.zeros(X_test.shape[0])
    feat_importance_df = pd.DataFrame()

    for n_fold, (trn_idx, val_idx) in enumerate(folds_idx):
        trn_X, trn_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        val_X, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]

        clf = LGBMClassifier(
            n_estimators=100000,
            learning_rate=0.008,
            num_leaves=30,
            colsample_bytree=.04,
            subsample=0.3,
            max_depth=-1,
            min_child_weight=10.0,
            min_child_samples=80,
            silent=-1,
            verbose=-1,
        )

        clf.fit(trn_X, trn_y, eval_set=[(trn_X, trn_y), (val_X, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=100)

        print("best iteration: ", clf.best_iteration_)
        y_pred_train[val_idx] = clf.predict_proba(val_X, num_iteration=clf.best_iteration_)[:,1]
        y_pred_test += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1] / N_SPLIT

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feat_importance_df = pd.concat([feat_importance_df, fold_importance_df], axis=0)

        print("Fold {:d} AUC : {:.3f}".format(n_fold + 1, roc_auc_score(val_y, y_pred_train[val_idx])))
        del clf, trn_X, trn_y, val_X, val_y
        gc.collect()

    print('Full validation AUC score {:.3f}'.format(roc_auc_score(y_train, y_pred_train)))
    print('Test AUC score {:.3f}'.format(roc_auc_score(y_test, y_pred_test)))

    return y_pred_train, y_pred_test, feat_importance_df

def display_roc_curve(y_train, y_test, y_pred_train, y_pred_test, folds_idx):

    plt.figure(figsize=(7,7))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx):

        fpr, tpr, thresholds = roc_curve(y_train.iloc[val_idx], y_pred_train[val_idx])
        score = roc_auc_score(y_train.iloc[val_idx], y_pred_train[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                 label='ROC fold {:d}(AUC = {:.3f})'.format(n_fold + 1, score))


    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    score = roc_auc_score(y_test, y_pred_test)
    plt.plot(fpr, tpr, lw=2, alpha=0.7, color='r',
             label='ROC test data (AUC = {:.3f})'.format(score))

        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Blind luck', alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('roc_curve.png')

def display_importance(feat_importance_df):

    cols = feat_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feat_importance_df.loc[feat_importance_df.feature.isin(cols)]
    
    plt.figure(figsize=(9,8))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = build_input()

    folds = KFold(n_splits=N_SPLIT, shuffle=True, random_state=42)
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(X_train)]

    y_pred_train, y_pred_test, feat_importance_df = \
        train_model(X_train, X_test, y_train, y_test, folds_idx)

    display_roc_curve(y_train, y_test, y_pred_train, y_pred_test, folds_idx)
    display_importance(feat_importance_df)
    plt.show()

