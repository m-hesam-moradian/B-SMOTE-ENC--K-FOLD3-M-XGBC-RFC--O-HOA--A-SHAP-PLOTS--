import pandas as pd
from imblearn.over_sampling import SMOTENC


def balance_with_smote_enc(file_path, target_col, categorical_cols, output_path=None):
    """
    Balances a dataset using SMOTE-ENC for categorical + numeric features.

    Args:
        file_path (str): Path to the CSV file.
        target_col (str): Name of the target column.
        categorical_cols (list): List of categorical column names.
        output_path (str, optional): If given, saves balanced dataset to this path.

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    # Load CSV
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Get indices of categorical columns
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

    # Encode categorical columns to numeric codes
    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes

    # Apply SMOTENC
    smote_enc = SMOTENC(categorical_features=categorical_indices, random_state=42)
    X_resampled, y_resampled = smote_enc.fit_resample(X_encoded, y)

    # Decode categorical columns back to original labels
    for col in categorical_cols:
        X_resampled[col] = pd.Categorical.from_codes(
            X_resampled[col], categories=X[col].astype("category").cat.categories
        )

    # Combine back into a DataFrame
    balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

    # Save if output_path provided
    if output_path:
        balanced_df.to_csv(output_path, index=False)

    print("Before balancing:\n", y.value_counts())
    print("\nAfter balancing:\n", y_resampled.value_counts())

    return balanced_df
