import polars as pl
import pytest
from polars.testing import assert_frame_equal

from casie.utils import update_indexed_frame


@pytest.fixture(params=["features", "variants"])
def indexed_frame(request):
    if request.param == "features":
        # A feature can occur in multiple collections; identity is the pair.
        frame = pl.DataFrame(
            {
                "_collection_idx": pl.Series([0, 0, 1], dtype=pl.UInt32),
                "_pgs_idx": pl.Series([8, 3, 9], dtype=pl.UInt32),
                "collection_id": ["first", "first", "second"],
                "feature_id": ["gene1", "gene2", "gene1"],
                "annotation": ["old1", "old2", "old3"],
            }
        )
        return frame, ["collection_id", "feature_id"], ["_collection_idx", "_pgs_idx"]

    frame = pl.DataFrame(
        {
            "_variant_idx": pl.Series([8, 3, 9], dtype=pl.UInt32),
            "variant_id": ["1:10:A:G", "1:20:C:T", "2:10:A:C"],
            "annotation": ["old1", "old2", "old3"],
        }
    )
    return frame, ["variant_id"], ["_variant_idx"]


@pytest.mark.parametrize("n_rows", [3, 2], ids=["equal", "subset"])
def test_updates_accept_existing_keys_and_preserve_order_and_indices(
    indexed_frame, n_rows
):
    old, keys, indices = indexed_frame
    original = old.clone()
    expected = old.reverse().head(n_rows).drop("annotation").with_columns(
        label=pl.lit("new")
    )
    new = expected.drop(indices)
    original_new = new.clone()

    result = update_indexed_frame(old, new, indices=indices, keys=keys)

    assert_frame_equal(result, expected)
    assert_frame_equal(old, original)
    assert_frame_equal(new, original_new)


def test_updates_accept_unchanged_frames(indexed_frame):
    old, keys, indices = indexed_frame

    result = update_indexed_frame(old, old.drop(indices), indices=indices, keys=keys)

    assert_frame_equal(result, old)


def test_updates_accept_annotation_joins(indexed_frame):
    old, keys, indices = indexed_frame
    annotations = old.select(keys).reverse().with_columns(label=pl.lit("new"))
    new = old.drop(indices).join(
        annotations, on=keys, how="left", maintain_order="left"
    )

    result = update_indexed_frame(old, new, indices=indices, keys=keys)

    assert_frame_equal(result, old.with_columns(label=pl.lit("new")))


def test_updates_reject_unknown_keys(indexed_frame):
    old, keys, indices = indexed_frame
    original = old.clone()
    new = old.drop(indices).head(1).with_columns(pl.lit("unknown").alias(keys[-1]))

    with pytest.raises(ValueError, match="must be a subset of the existing set"):
        update_indexed_frame(old, new, indices=indices, keys=keys)

    assert_frame_equal(old, original)


def test_updates_validate_complete_key_tuples():
    old = pl.DataFrame(
        {
            "_pgs_idx": [0, 1],
            "collection_id": ["first", "second"],
            "feature_id": ["gene1", "gene2"],
        }
    )
    # Both scalar values exist, but their combination does not.
    new = pl.DataFrame({"collection_id": ["first"], "feature_id": ["gene2"]})

    with pytest.raises(ValueError, match="must be a subset of the existing set"):
        update_indexed_frame(
            old, new, indices=["_pgs_idx"], keys=["collection_id", "feature_id"]
        )


@pytest.mark.parametrize("invalid", ["empty", "missing", "null", "duplicate", "protected"])
def test_updates_reject_invalid_keys_and_supplied_indices(indexed_frame, invalid):
    old, keys, indices = indexed_frame
    original = old.clone()
    new = old.drop(indices)
    if invalid == "empty":
        new = new.head(0)
        message = "New metadata is empty"
    elif invalid == "missing":
        new = new.drop(keys[0])
        message = "Missing identity columns"
    elif invalid == "null":
        new = new.with_columns(pl.lit(None, dtype=pl.String).alias(keys[0]))
        message = "Identity columns.*missing values"
    elif invalid == "duplicate":
        new = pl.concat([new, new.head(1)])
        message = "Duplicates in identity columns"
    else:
        new = new.with_columns(pl.lit(0).alias(indices[0]))
        message = "protected from modification"

    with pytest.raises(ValueError, match=message):
        update_indexed_frame(old, new, indices=indices, keys=keys)

    assert_frame_equal(old, original)


def test_subset_validation_does_not_depend_on_hidden_indices():
    old = pl.DataFrame({"feature_id": ["gene1"]})
    new = pl.DataFrame({"feature_id": ["gene2"]})

    with pytest.raises(ValueError, match="must be a subset of the existing set"):
        update_indexed_frame(old, new, indices=[], keys=["feature_id"])
