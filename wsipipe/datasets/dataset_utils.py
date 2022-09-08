import pandas as pd


def sample_dataset(df: pd.DataFrame, samples_per_class: str) -> pd.DataFrame:
    """ Create a subset of a dataset dataframe
    This function will create a smaller dataframe that only includes
    n slides per class. This can be used to create smaller datasets for
    example for debugging pipelines

    Args:
        df (pd.DataFrame): A dataframe containing a column called label
        samples_per_class: The number of slides per class to return
    Returns:
        df (pd.DataFrame): A copy of the dataframe with samples_per_class rows
            for each label
    """
    g = df.groupby("label")
    assert samples_per_class <= g.size().min(), f"Not enough samples for one of the classes. {samples_per_class} {g.size().min()}"
    def sample_group(x):
        return x.sample(samples_per_class).reset_index(drop=True) # todo: make this deterministic
    sampled = g.apply(sample_group)
    return sampled