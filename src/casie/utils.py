import polars as pl


def update_indexed_frame(
    old: pl.DataFrame,
    new: pl.DataFrame,
    indices: list[str],
    keys: list[str],
) -> pl.DataFrame:
    """Helper function for updating metadata and ensuring that indices
    and identity columns are preserved.

    Parameters
    ----------
    old : polars.DataFrame
        Dataframe with the old metadata
    new : polars.DataFrame
        Dataframe with the new metadata
    indices : list[str]
        Column names of indices which are hidden from the user
    keys : list[str]
        Names of identifier columns used for joining the new metadata

    Returns
    -------
    polars.DataFrame
        Updated and validated metadata

    Notes
    -----
    New key tuples must be unique, non-null, and a subset of the old key
    tuples; equality is allowed, but empty frames are rejected. Rows are matched
    by their supplied keys, irrespective of their order. Non-key columns can be added,
    changed, or dropped, but hidden indices cannot be supplied by the caller.
    The old frame must have unique keys and non-null hidden indices.
    """
    if new.is_empty():
        raise ValueError("New metadata is empty")

    for col in indices:
        if col in new.columns:
            raise ValueError(f"Column {col} is protected from modification")

    missing = set(keys) - set(new.columns)
    if missing:
        raise ValueError(f"Missing identity columns: {sorted(missing)}")

    if any(new[key].null_count() for key in keys):
        raise ValueError(f"Identity columns {keys} contain missing values")

    if new.select(keys).is_duplicated().any():
        raise ValueError(f"Duplicates in identity columns {keys}")

    unknown_keys = new.select(keys).join(
        old.select(keys),
        on=keys,
        how="anti",
        maintain_order="left",
    )
    if not unknown_keys.is_empty():
        raise ValueError(
            f"Values in identity columns {keys} must be a subset of the existing set"
        )

    update = old.select(*indices, *keys).join(
        new,
        on=keys,
        how="right",
        coalesce=True,
        maintain_order="right",
    )
    if any(update[col].null_count() for col in indices):
        raise ValueError("Only existing rows can be updated")

    return update
