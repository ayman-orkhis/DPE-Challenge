from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# The submission here should simply be a function that returns a model
# compatible with scikit-learn API (must have fit and predict methods).
#
# This baseline uses a Pipeline that:
# 1. Encodes categorical (string) columns with OrdinalEncoder
# 2. Passes numeric columns through unchanged
# 3. Fits a RandomForestRegressor
def get_model():

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1),
             lambda X: X.select_dtypes(include=["object", "category"]).columns.tolist()),
        ],
        remainder="passthrough",  # keep numeric columns as-is
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, random_state=42
        )),
    ])

    return model
