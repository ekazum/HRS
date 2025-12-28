# Guesser Model Dimension Mismatch Fix

## Problem Description

When running `guesser_main`, users encountered the following error:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3669 and 714x64)
```

This error occurred in the test function when trying to process test data through a loaded model.

## Root Cause

The error was caused by a mismatch between the saved model's architecture and the current data structure:

1. A model was previously trained and saved with data that produced **714 input features**
2. When running the script again, the data structure changed (or different data/configuration was used)
3. The new data produces **3669 input features**
4. When the test function tries to load the old saved model and process new data, the dimensions don't match

### Why This Happens

The `MultimodalGuesser` model calculates the number of input features (`features_total`) based on the data during initialization. This calculation depends on:

- Number of features in the dataset
- Types of features (text, numeric, images, time series)
- Feature mapping strategy

When the data structure changes between training runs (different dataset, different preprocessing, different configuration), the `features_total` can change, causing the saved model to be incompatible.

## Solution

The fix adds validation and clear error handling in the `test()` function:

### 1. File Existence Check
```python
if not os.path.exists(guesser_load_path):
    raise FileNotFoundError(...)
```

### 2. State Dict Loading with Error Handling
```python
try:
    model.load_state_dict(guesser_state_dict, strict=True)
except RuntimeError as e:
    if "size mismatch" in str(e):
        raise RuntimeError(
            "The saved model architecture does not match the current model. "
            "This typically happens when the data structure or configuration has changed. "
            ...
        )
```

### 3. Forward Pass Validation
```python
if len(X_test) > 0:
    try:
        test_input = X_test[0]
        _ = model(test_input)
    except RuntimeError as e:
        if "mat1 and mat2 shapes cannot be multiplied" in str(e):
            raise RuntimeError(
                "The loaded model cannot process the test data due to dimension mismatch. "
                ...
            )
```

## How to Fix the Error

If you encounter this error, follow these steps:

1. **Locate the saved model file**: The error message will show the path to the saved model (e.g., `./saved_models/best_guesser.pth`)

2. **Delete the old saved model**:
   ```bash
   rm ./saved_models/best_guesser.pth
   ```
   Or on Windows:
   ```cmd
   del saved_models\best_guesser.pth
   ```

3. **Re-run the training**: Execute the guesser_main script again:
   ```bash
   python -m src.Guesser.guesser_main
   ```

The script will:
- Initialize the model with the current data structure
- Train the model from scratch
- Save the new model with the correct architecture
- Test the model successfully

## Prevention

To avoid this issue in the future:

1. **Use consistent data**: Ensure you're using the same data and preprocessing pipeline between training runs

2. **Clear old models**: Delete saved models when switching datasets or changing model configuration

3. **Version your models**: Include dataset version or configuration in the model filename:
   ```python
   guesser_filename = f'best_guesser_{dataset_version}.pth'
   ```

4. **Save metadata**: Consider saving model configuration alongside weights:
   ```python
   torch.save({
       'state_dict': model.state_dict(),
       'features_total': model.features_total,
       'map_feature': model.map_feature,
       'config': FLAGS.__dict__
   }, save_path)
   ```

## Technical Details

### Model Architecture Dependency

The model's first layer dimensions depend on `features_total`:
```python
self.layer1 = torch.nn.Sequential(
    torch.nn.Linear(self.features_total, FLAGS.hidden_dim1),
    ...
)
```

`features_total` is calculated by `map_features_to_indices()` which analyzes the dataset to determine:
- How many features exist
- Whether each feature is text (20 dimensions), numeric (1 dimension), or image (20 dimensions)

### Forward Pass Dimension Calculation

During forward pass, each sample's features are embedded:
- Text features → `text_reduced_dim` dimensions (e.g., 20)
- Image features → 20 dimensions
- Numeric features → 1 dimension
- NaN features → Size based on `map_feature` dictionary

These embeddings are concatenated to form the input to `layer1`, which must match `features_total`.
