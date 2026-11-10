import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

from features import build_datasets, TARGET, FEATURES


XGB_PARAMS = {
    'eta': 0.05,
    'max_depth': 4,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

NUM_BOOST_ROUND = 117

OUTPUT_FILE = Path(__file__).parent / "model.bin"


def train_model(model_df):

    train_dicts = model_df[FEATURES].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train_dicts)
    y_train = model_df[TARGET].values

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)

    return dv, model



if __name__ == "__main__":

    print("Building datasets...")
    model_df, snapshot = build_datasets()

    print("Training model...")
    dv, model = train_model(model_df)
    print("Model trained")

    print("Saving model")
    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model, snapshot), f_out)

    print(f"Model saved to {OUTPUT_FILE}")