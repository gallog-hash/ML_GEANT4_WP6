# src/utils/data_loader_utils.py

def create_data_loaders(
    train_data=None,
    val_data=None,
    test_data=None,
    batch_size: int = 128,
    shuffle: dict = None,
    data_type: dict = None,
) -> dict:
    """
    Create DataLoaders for any combination of train, validation, and test
    datasets. 
    
    Args:
        train_data (Dataset, optional): Dataset for the training split. Defaults
            to None.
        val_data (Dataset, optional): Dataset for the validation split. Defaults
            to None.
        test_data (Dataset, optional): Dataset for the test split. Defaults to
            None.
        batch_size (int, optional): Batch size used for all DataLoaders.
            Defaults to 128.
        shuffle (dict, optional): Dictionary specifying whether to shuffle the  
            data for each split. Keys should be "train", "val", and "test", with
            boolean values. Defaults to {"train": True, "val": False, "test":
            False}.
        data_type (dict, optional): Dictionary specifying the type of data for
            each split. Keys should be "train", "val", and "test", with string
            values. Defaults to {"train": "training", "val": "validation",
            "test": "test"}. 
    
    Returns:
        dict: A dictionary with DataLoader objects for requested splits.
    """
    from core.preprocessing.data_utils import create_data_loader

    shuffle = shuffle or {"train": True, "val": False, "test": False}
    data_type = data_type or {
        "train": "training", "val": "validation", "test": "test"
        }
    
    valid_keys = {"train", "val", "test"}

    # Validate shuffle keys
    if not set(shuffle.keys()).issubset(valid_keys):
        invalid = set(shuffle.keys()) - valid_keys
        raise ValueError(
            f"Invalid shuffle keys: {invalid}. Must be any of {valid_keys}."
        )

    # Validate data_type keys
    if not set(data_type.keys()).issubset(valid_keys):
        invalid = set(data_type.keys()) - valid_keys
        raise ValueError(
            f"Invalid data_type keys: {invalid}. Must be any of {valid_keys}."
        )
    
    loaders = {}

    if train_data is not None:
        loaders["train"] = create_data_loader(
            data=train_data,
            data_type=data_type.get("train", "training"),
            batch_size=batch_size,
            shuffle=shuffle.get("train", True)
        )

    if val_data is not None:
        loaders["val"] = create_data_loader(
            data=val_data,
            data_type=data_type.get("val", "validation"),
            batch_size=batch_size,
            shuffle=shuffle.get("val", False)
        )

    if test_data is not None:
        loaders["test"] = create_data_loader(
            data=test_data,
            data_type=data_type.get("test", "test"),
            batch_size=batch_size,
            shuffle=shuffle.get("test", False)
        )

    return loaders
