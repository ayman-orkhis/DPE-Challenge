from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# The submission here should simply be a function that returns a model
# compatible with scikit-learn API (must have fit and predict methods).
#
# This baseline uses a Pipeline that:
# 1. Encodes categorical (string) columns with OrdinalEncoder
# 2. Passes numeric columns through unchanged
# 3. Fits a RandomForestRegressor
def get_model():

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_pipeline,
                lambda X: X.select_dtypes(include=["object", "category"]).columns.tolist()
            ),
            (
                "num",
                numeric_pipeline,
                lambda X: X.select_dtypes(exclude=["object", "category"]).columns.tolist()
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )),
    ])

    return model
