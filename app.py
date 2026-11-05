from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = PROJECT_DIR / "train.csv"
DEFAULT_TEST_FILE = PROJECT_DIR / "test.csv"

REQUIRED_TRAIN_COLUMNS = {
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
}
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def read_csv(file_bytes: bytes, description: str) -> pd.DataFrame:
    """Read a user-provided CSV and explain common file-format errors."""
    try:
        data = pd.read_csv(BytesIO(file_bytes))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read the {description} CSV. Check that it is valid CSV.") from error
    data.columns = data.columns.str.strip()
    return data


def prepare_features(passengers: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs from passenger facts; discard IDs and raw name text."""
    missing = {"Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"} - set(
        passengers.columns
    )
    if missing:
        raise ValueError("Passenger data is missing columns: " + ", ".join(sorted(missing)))

    features = passengers[
        ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].copy()
    features["Title"] = (
        features["Name"]
        .fillna("")
        .astype(str)
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
    )
    features["Title"] = features["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    common_titles = {"Mr", "Mrs", "Miss", "Master"}
    features["Title"] = features["Title"].where(
        features["Title"].isin(common_titles), "Other"
    )
    features["FamilySize"] = (
        pd.to_numeric(features["SibSp"], errors="coerce").fillna(0)
        + pd.to_numeric(features["Parch"], errors="coerce").fillna(0)
        + 1
    )
    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype(str).str.strip()
        features[column] = features[column].replace("", "Unknown")
    return features[FEATURE_NAMES]


def build_model(number_of_trees: int, maximum_depth: int) -> Pipeline:
    """Build a complete preprocessing and prediction pipeline."""
    numeric_steps = Pipeline(
        steps=[("fill_missing", SimpleImputer(strategy="median"))]
    )
    categorical_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode_categories", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numbers", numeric_steps, NUMERIC_FEATURES),
            ("categories", categorical_steps, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=number_of_trees,
        max_depth=maximum_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocess", preprocessing), ("classifier", classifier)]
    )


@st.cache_resource(show_spinner="Training the model and checking its results...")
def train_model(
    train_bytes: bytes, number_of_trees: int, maximum_depth: int
) -> dict[str, Any]:
    """Evaluate on held-out passengers, then fit a final model on all training rows."""
    training_data = read_csv(train_bytes, "training")
    missing = REQUIRED_TRAIN_COLUMNS - set(training_data.columns)
    if missing:
        raise ValueError(
            "Training CSV is missing required columns: " + ", ".join(sorted(missing))
        )
    if len(training_data) < 10:
        raise ValueError("Training data must contain at least 10 passenger rows.")

    labels = pd.to_numeric(training_data["Survived"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Survived must contain only 0 (did not survive) or 1 (survived).")
    labels = labels.astype(int)
    if labels.value_counts().min() < 2:
        raise ValueError("Training data needs at least two passengers in each outcome group.")

    features = prepare_features(training_data)
    train_x, valid_x, train_y, valid_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    validation_model = build_model(number_of_trees, maximum_depth)
    validation_model.fit(train_x, train_y)
    predictions = validation_model.predict(valid_x)
    probabilities = validation_model.predict_proba(valid_x)[:, 1]

    final_model = build_model(number_of_trees, maximum_depth)
    final_model.fit(features, labels)
    transformed_names = final_model.named_steps["preprocess"].get_feature_names_out()
    importances = final_model.named_steps["classifier"].feature_importances_
    feature_importance = (
        pd.DataFrame({"Feature": transformed_names, "Importance": importances})
        .assign(Feature=lambda frame: frame["Feature"].str.replace(
            r"^(numbers|categories)__", "", regex=True
        ))
        .groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    return {
        "model": final_model,
        "row_count": len(training_data),
        "survival_rate": float(labels.mean()),
        "metrics": {
            "accuracy": float(accuracy_score(valid_y, predictions)),
            "precision": float(precision_score(valid_y, predictions, zero_division=0)),
            "recall": float(recall_score(valid_y, predictions, zero_division=0)),
            "f1": float(f1_score(valid_y, predictions, zero_division=0)),
            "mean_survival_probability": float(probabilities.mean()),
        },
        "confusion_matrix": confusion_matrix(valid_y, predictions, labels=[0, 1]),
        "feature_importance": feature_importance,
        "validation_count": len(valid_y),
    }


def main() -> None:
    st.set_page_config(page_title="Titanic Survival Explorer", page_icon="🚢", layout="wide")
    st.title("Titanic Survival Explorer")
    st.write(
        "Train a machine-learning model on Titanic passenger records, check how it does "
        "on passengers it has not seen, and try your own passenger details."
    )

    with st.sidebar:
        st.header("1. Choose your data")
        uploaded_training = st.file_uploader(
            "Training CSV (Kaggle train.csv)", type=["csv"]
        )
        uploaded_test = st.file_uploader(
            "Optional passenger CSV (Kaggle test.csv)", type=["csv"]
        )
        if DEFAULT_TRAIN_FILE.exists() and not uploaded_training:
            st.caption("Using the project's train.csv. Upload another file to replace it.")

        st.header("2. Model settings")
        number_of_trees = st.slider(
            "Number of decision trees", min_value=50, max_value=500, value=200, step=50
        )
        maximum_depth = st.slider(
            "Tree depth", min_value=2, max_value=16, value=6,
            help="Smaller trees are simpler. Very deep trees can memorize the training data.",
        )

    if uploaded_training:
        training_bytes = uploaded_training.getvalue()
    elif DEFAULT_TRAIN_FILE.exists():
        training_bytes = DEFAULT_TRAIN_FILE.read_bytes()
    else:
        training_bytes = b""

    if not training_bytes:
        st.info(
            "To get started, download **train.csv** and **test.csv** from the Kaggle "
            "Titanic competition, then upload train.csv in the sidebar. The training file "
            "must include the `Survived` column."
        )
        st.link_button("Open the Kaggle Titanic data page", "https://www.kaggle.com/competitions/titanic/data")
        st.stop()

    try:
        result = train_model(training_bytes, number_of_trees, maximum_depth)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not train the model: {error}")
        st.stop()

    st.success(
        f"Model trained on {result['row_count']:,} passenger records. "
        f"The evaluation uses a separate holdout of {result['validation_count']:,} passengers."
    )

    overview_tab, predict_tab, test_tab, guide_tab = st.tabs(
        ["Model results", "Try a passenger", "Predict a CSV", "How it works"]
    )

    with overview_tab:
        st.subheader("How well did it do on unseen passengers?")
        st.caption(
            "A random 20% of the training records is held back for this check. "
            "These results are a useful estimate, not a guarantee on new passengers."
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{result['metrics']['accuracy']:.1%}")
        metric_columns[1].metric("Precision", f"{result['metrics']['precision']:.1%}")
        metric_columns[2].metric("Recall", f"{result['metrics']['recall']:.1%}")
        metric_columns[3].metric("F1 score", f"{result['metrics']['f1']:.1%}")

        st.markdown(
            "- **Accuracy:** share of holdout passengers classified correctly.\n"
            "- **Precision:** of passengers predicted to survive, how many did survive?\n"
            "- **Recall:** of passengers who survived, how many did the model find?\n"
            "- **F1:** a balance between precision and recall."
        )

        st.subheader("Prediction check")
        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actually did not survive", "Actually survived"],
            columns=["Predicted not survive", "Predicted survive"],
        )
        st.dataframe(matrix, width="stretch")
        st.caption(
            "Rows are the real outcomes; columns are the model's predictions. "
            "The diagonal cells are correct predictions."
        )

        st.subheader("Which details influenced the model most?")
        st.caption(
            "These are relative Random Forest feature-importance scores, not proof that "
            "a passenger detail caused an outcome."
        )
        st.dataframe(
            result["feature_importance"].style.format({"Importance": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )

    with predict_tab:
        st.subheader("Try an example passenger")
        st.write(
            "Enter passenger details below. The model returns a probability, not a "
            "certain historical outcome."
        )
        with st.form("passenger_form"):
            name = st.text_input(
                "Passenger name",
                value="Example, Mr. Passenger",
                help="A title such as Mr., Mrs., Miss., or Master. helps the model use the name.",
            )
            field_columns = st.columns(3)
            passenger_class = field_columns[0].selectbox("Passenger class", [1, 2, 3], index=2)
            sex = field_columns[1].selectbox("Sex", ["female", "male"])
            age = field_columns[2].number_input(
                "Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0
            )
            sib_sp = field_columns[0].number_input(
                "Siblings / spouse aboard", min_value=0, max_value=10, value=0
            )
            parch = field_columns[1].number_input(
                "Parents / children aboard", min_value=0, max_value=10, value=0
            )
            fare = field_columns[2].number_input(
                "Ticket fare", min_value=0.0, max_value=1000.0, value=32.0, step=1.0
            )
            embarked = st.selectbox(
                "Port of embarkation", ["S", "C", "Q", "Unknown"],
                format_func=lambda code: {
                    "S": "Southampton (S)",
                    "C": "Cherbourg (C)",
                    "Q": "Queenstown (Q)",
                    "Unknown": "Unknown",
                }[code],
            )
            submitted = st.form_submit_button("Estimate survival chance")

        if submitted:
            passenger = pd.DataFrame(
                [
                    {
                        "Name": name,
                        "Pclass": passenger_class,
                        "Sex": sex,
                        "Age": age,
                        "SibSp": sib_sp,
                        "Parch": parch,
                        "Fare": fare,
                        "Embarked": embarked,
                    }
                ]
            )
            probability = float(
                result["model"].predict_proba(prepare_features(passenger))[0, 1]
            )
            st.metric("Estimated survival probability", f"{probability:.1%}")
            if probability >= 0.5:
                st.success("The model predicts: **more likely to survive**.")
            else:
                st.info("The model predicts: **more likely not to survive**.")
            st.caption(
                "This is a pattern learned from historical data, not a statement of "
                "certainty or a judgment about any individual."
            )

    with test_tab:
        st.subheader("Make predictions for a Kaggle test file")
        st.write(
            "Upload Kaggle's test.csv to predict the passengers whose survival labels "
            "are hidden. Download the results as a submission file."
        )
        if uploaded_test:
            test_bytes = uploaded_test.getvalue()
        elif DEFAULT_TEST_FILE.exists():
            test_bytes = DEFAULT_TEST_FILE.read_bytes()
            st.caption("Using the project's test.csv. Upload another file in the sidebar to replace it.")
        else:
            test_bytes = b""

        if not test_bytes:
            st.info("Upload test.csv in the sidebar to generate passenger predictions.")
        else:
            try:
                test_data = read_csv(test_bytes, "test")
                test_features = prepare_features(test_data)
                test_predictions = result["model"].predict(test_features).astype(int)
                passenger_ids = (
                    test_data["PassengerId"]
                    if "PassengerId" in test_data.columns
                    else pd.Series(range(1, len(test_data) + 1), name="PassengerId")
                )
                submission = pd.DataFrame(
                    {"PassengerId": passenger_ids, "Survived": test_predictions}
                )
                st.success(f"Generated predictions for {len(submission):,} passengers.")
                st.dataframe(submission.head(20), width="stretch", hide_index=True)
                st.download_button(
                    "Download Kaggle predictions CSV",
                    submission.to_csv(index=False).encode("utf-8"),
                    file_name="titanic_predictions.csv",
                    mime="text/csv",
                )
            except (ValueError, KeyError) as error:
                st.error(f"Could not predict this file: {error}")

    with guide_tab:
        st.subheader("What does the app do?")
        st.markdown(
            "1. **Loads passenger records.** The training CSV needs a `Survived` column "
            "(`0` = did not survive, `1` = survived).\n"
            "2. **Prepares the details.** Missing ages and fares are filled with typical "
            "training-set values; text categories are converted into model-readable values.\n"
            "3. **Creates helpful features.** The model uses passenger class, sex, age, "
            "ticket fare, family size, whether someone traveled alone, embarkation port, "
            "and a title extracted from the name.\n"
            "4. **Checks its work.** A stratified 20% holdout is used for metrics; the "
            "final model is then trained using all labeled training rows.\n"
            "5. **Makes predictions.** Enter one example or upload Kaggle's test.csv."
        )
        st.info(
            "Passenger IDs and raw ticket numbers are not used as predictive clues. "
            "The raw name is only used to extract a title. This educational model can "
            "reflect historical bias and should not be used to make real-world decisions."
        )
        st.write(
            f"The current training file's observed survival rate is "
            f"**{result['survival_rate']:.1%}**. The app uses a Random Forest classifier "
            "with a fixed random seed, so the same data and settings produce repeatable results."
        )
        st.link_button(
            "View the Kaggle Titanic competition",
            "https://www.kaggle.com/competitions/titanic/data",
        )


if __name__ == "__main__":
    main()
