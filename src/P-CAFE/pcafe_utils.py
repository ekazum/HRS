import os
from sklearn.datasets import fetch_openml
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy


def balance_class(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, add_noise(X[noisy_indices], noise_std)], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced




def map_multiple_features(sample):
    index_map = {}
    for i in range(0, sample.shape[0]):
        index_map[i] = [i]
    return index_map


def map_multiple_features_for_logistic_mimic(sample):
    # map the index features that each test reveals
    index_map = {}
    for i in range(0, 17):
        index_map[i] = list(range(i * 42, i * 42 + 42))
    return index_map

def map_time_series(sample):
    """
    Map features to indices based on their type.
    If any value in a column is a string, treat the column as text.
    :param sample: 2D array-like dataset (list of lists, numpy array, or pandas DataFrame)
    :return: A dictionary mapping feature indices to lists of indices and the total number of features
    """
    index_map = {}
    current_index = 0
    for col_index in range(sample.shape[1]+1):
            if col_index == 0:
                index_map[col_index] = list(range(current_index, current_index + 20))
                current_index += 20
            else:
                index_map[col_index] = [current_index]  # Numeric feature
                current_index += 1
    return index_map


def load_mimic_time_series():
    # Get the current working directory
    base_dir = os.getcwd()
    # Construct file paths dynamically
    #train_X_path = os.path.join(base_dir, 'input\\data_time_series\\train_X.csv')
    #train_Y_path = os.path.join(base_dir, 'input\\data_time_series\\train_Y.csv')
    #val_X_path = os.path.join(base_dir, 'input\\data_time_series\\val_X.csv')
    #val_Y_path = os.path.join(base_dir, 'input\\data_time_series\\val_Y.csv')
    #test_X_path = os.path.join(base_dir, 'input\\data_time_series\\test_X.csv')
    #test_Y_path = os.path.join(base_dir, 'input\\data_time_series\\test_Y.csv')
    train_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\train_X.csv')
    train_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\train_Y.csv')
    val_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\val_X.csv')
    val_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\val_Y.csv')
    test_X_path = os.path.join(base_dir, 'data\\input\\data_time_series\\test_X.csv')
    test_Y_path = os.path.join(base_dir, 'data\\input\\data_time_series\\test_Y.csv')

    # Read the files
    X_train = pd.read_csv(train_X_path)
    Y_train = pd.read_csv(train_Y_path)
    X_val = pd.read_csv(val_X_path)
    Y_val = pd.read_csv(val_Y_path)
    X_test = pd.read_csv(test_X_path)
    Y_test = pd.read_csv(test_Y_path)

    X = pd.concat([X_train, X_val, X_test])
    Y = pd.concat([Y_train, Y_val, Y_test])
    Y = Y.to_numpy().reshape(-1)
    # balance classes no noise
    X, Y = balance_class_no_noise(X.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features_for_logistic_mimic(X.iloc[0])
    return X, Y, 17, map_test


def clean_mimic_data_nan(X):
    # Keeps columns with at least 20% non-missing values.
    X = X.dropna(axis=1, thresh=int(0.2 * X.shape[0]))
    return X


def reduce_number_of_samples(X, Y):
    # reduce the number of samples to 1000
    X = X[:1000]
    Y = Y[:1000]
    return X, Y



def df_to_list(df):
    grp_df = df.groupby('patientunitstayid')
    df_arr = []

    for idx, frame in grp_df:
        # Sort the dataframe for each patient based on 'itemoffset'
        sorted_frame = frame.sort_values(by='itemoffset', ascending=True)  # or True for ascending
        df_arr.append(sorted_frame)

    return df_arr


def load_time_series():
    base_dir = os.getcwd()
    path = os.path.join(base_dir, 'input\\df_data.csv')
    # read csv file from path
    df_time_series = pd.read_csv(path)
    df_time_series = df_to_list(df_time_series)
    labels = []
    patients = []
    for df in df_time_series:
        df = df.drop(df.columns[0:3], axis=1)
        # Drop the last column (assuming it's the label)
        labels.append(df.iloc[0, -1])
        df = df.iloc[:, :-1]
        df_history = df.iloc[:-1]  # all except last row

        history_stats = df_history.agg(['mean', 'std', 'min', 'max']).values.flatten()
        recent_values = df.iloc[[-1]].values.flatten()
        embedding = np.concatenate([recent_values, history_stats])
        patients.append(embedding)

    X, Y = balance_class_no_noise(np.array(patients), np.array(labels).reshape(-1))
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test



def load_time_Series():
    base_dir = os.getcwd()
    path = os.path.join(base_dir, 'input\\df_data.csv')
    # read csv file from path
    df_time_series = pd.read_csv(path)
    df_time_series = df_to_list(df_time_series)
    labels = []
    patients = []
    for df in df_time_series:
        df = df.drop(df.columns[0:3], axis=1)
        # Drop the last column (assuming it's the label)
        labels.append(df.iloc[0, -1])
        df = df.iloc[:, :-1]
        patients.append(df)
    patients, labels = balance_class_no_noise_dfs(patients, labels)

    map_test = map_time_series(patients[0])
    return patients, labels, len(patients[0].columns) + 1, map_test



def load_mimic_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_with_text.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id', 'los'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    X = clean_mimic_data_nan(X)
    map_test = map_multiple_features(X.iloc[0])
    # X,Y = reduce_number_of_samples(X,Y)
    return X, Y, len(X.columns), map_test


def load_mimic_no_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_numeric.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test


def load_mimic_only_text():
    base_dir = os.getcwd()
    # Construct file paths dynamically
    path = os.path.join(base_dir, 'input\\data_with_text.json')
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id'])
    # define the label mortality_inhospital as Y and drop from df
    Y = df['mortality_inhospital'].to_numpy().reshape(-1)
    df = df.drop(columns=['mortality_inhospital'])
    # balance classes no noise
    X, Y = balance_class_no_noise(df.to_numpy(), Y)
    X = pd.DataFrame(X)
    # keep only the text columns
    X = clean_mimic_data_nan(X)
    # take only str columns
    X = X.iloc[:, [3, 4, 5]]
    # print how many nan values in each column
    # nan_percentage = (X.isna().sum() / len(X)) * 100
    # print(nan_percentage)
    map_test = map_multiple_features(X.iloc[0])
    return X, Y, len(X.columns), map_test



def import_breast():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_prognostic.data.features.to_numpy()
    y = breast_cancer_wisconsin_prognostic.data.targets.to_numpy()
    y[y == 'R'] = 1
    y[y == 'N'] = 0
    X[X == 'nan'] = 0
    X = np.nan_to_num(X, nan=0)
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)

    return X, y, breast_cancer_wisconsin_prognostic.metadata.num_features, breast_cancer_wisconsin_prognostic.metadata.num_features




def load_image_data():
    # Fetch the MNIST dataset from openml
    mnist = fetch_openml('mnist_784', version=1)

    images = mnist.data.values.reshape(-1, 28, 28).astype(np.uint8)
    labels = mnist.target.astype(int)

    # Create a directory to store the images
    image_dir = Path("mnist_images")
    image_dir.mkdir(exist_ok=True)

    # Save images and store their paths in a list
    image_paths = []

    for i, (image, label) in enumerate(zip(images, labels)):
        # Include the label in the image path
        image_path = image_dir / f"label_{label}_image_{i}.png"
        Image.fromarray(image).save(image_path)
        image_paths.append(str(image_path))

    # Generate a random numeric feature
    numeric_feature = np.random.rand(len(image_paths))

    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'numeric_feature': numeric_feature,
        'label': labels
    })

    # Create numpy arrays for image paths and numeric features
    X = df[['image_path', 'numeric_feature']].values
    y = df['label'].values
    num_features = X.shape[1]

    return X, y, num_features





def create():
    # Set a random seed for reproducibility

    # Number of points to generate
    num_points = 100

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)

    # Create a scatter plot of the dataset with color-coded labels
    plt.scatter(x1_values, x2_values, c=labels, cmap='viridis', marker='o', label='Data Points')
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values))
    return x, labels, x1_values, 3


def balance_class_multi(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    # Calculate the difference in sample counts for each class
    count_diff = max_class_count - class_counts

    # Initialize arrays to store balanced data
    X_balanced = X.copy()
    y_balanced = y.copy()

    # Add noise to the features of the minority classes to balance the dataset
    for minority_class, diff in zip(unique_classes, count_diff):
        if diff > 0:
            # Get indices of samples belonging to the current minority class
            minority_indices = np.where(y == minority_class)[0]

            # Randomly sample indices from the minority class to add noise
            noisy_indices = np.random.choice(minority_indices, diff, replace=True)

            # Add noise to the features of the selected samples
            X_balanced = np.concatenate([X_balanced, add_noise(X[noisy_indices], noise_std)], axis=0)
            y_balanced = np.concatenate([y_balanced, y[noisy_indices]], axis=0)

    return X_balanced, y_balanced


def create_n_dim():
    # Number of points to generate
    num_points = 2000

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)
    # create numpy of zeros
    X = np.zeros((num_points, 10))
    i = 0
    while i < num_points:
        # choose random index to assign x1 and x2 values
        index = np.random.randint(0, 10)
        # assign x1 to index for 5 samples
        X[i][index] = x1_values[i]
        X[i + 1][index] = x1_values[i + 1]
        X[i + 2][index] = x1_values[i + 2]
        X[i + 3][index] = x1_values[i + 3]
        X[i + 4][index] = x1_values[i + 4]
        # choose random index to assign x2 that is not the same as x1
        index2 = np.random.randint(0, 10)
        while index2 == index:
            index2 = np.random.randint(0, 10)
        X[i][index2] = x2_values[i]
        X[i + 1][index2] = x2_values[i + 1]
        X[i + 2][index2] = x2_values[i + 2]
        X[i + 3][index2] = x2_values[i + 3]
        X[i + 4][index2] = x2_values[i + 4]
        i += 5
    question_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return X, labels, question_names, 10





def balance_class_no_noise_dfs(patients, labels):
    """
    Balance classes by duplicating minority class patient time series (no noise added).

    Parameters:
    - patients: List of pandas DataFrames (one per patient).
    - labels: List or array of labels.

    Returns:
    - patients_balanced: List of balanced DataFrames.
    - labels_balanced: Balanced label list.
    """
    from collections import Counter
    import numpy as np

    # Count class instances
    class_counts = Counter(labels)
    unique_classes = list(class_counts.keys())
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Indices of each class
    majority_indices = [i for i, y in enumerate(labels) if y == majority_class]
    minority_indices = [i for i, y in enumerate(labels) if y == minority_class]

    # Determine how many more samples are needed
    diff = len(majority_indices) - len(minority_indices)

    if diff > 0:
        # Randomly sample with replacement from the minority class
        sampled_indices = np.random.choice(minority_indices, diff, replace=True)
        patients_balanced = patients + [patients[i].copy() for i in sampled_indices]
        labels_balanced = labels + [labels[i] for i in sampled_indices]
    else:
        patients_balanced = patients.copy()
        labels_balanced = labels.copy()

    return patients_balanced, labels_balanced

def balance_class_no_noise(X, y):
    """
    Balance classes by adding noise to the minority class's numeric features.

    Parameters:
    - X: Input features (numpy array with mixed data types).
    - y: Labels.
    - noise_std: Standard deviation of Gaussian noise.
    - numeric_columns: List or array of boolean values indicating numeric columns.

    Returns:
    - X_balanced: Balanced feature set.
    - y_balanced: Balanced labels.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, X[noisy_indices]], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()

    return X_balanced, y_balanced




