import sqlite3

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal
from scipy.sparse import csc_array

from casie.build import GenomeBuild
from casie.pgs import PGSCollection


@pytest.fixture
def weights():
    # Deliberately unsorted; variants are shared by two features.
    return pl.DataFrame(
        {
            "feature_id": ["gene2", "gene1", "gene2", "gene1"],
            "chr_name": ["chr2", "chr1", "1", "2"],
            "chr_position": ["20", "010", "2", "20"],
            "other_allele": ["C", "A", "C", "C"],
            "effect_allele": ["T", "G", "CA", "T"],
            "weight": [0.75, -0.5, 1.25, 0.125],
            "rsid": ["rs20", "rs10", "rs2", "rs20"],
        }
    )


def from_weights(weights, **kwargs):
    options = {
        "standardized": False,
        "genome_build": "hg38",
        "collection_id": "models",
        "feature_id": "feature_id",
        "effect_weight": "weight",
    }
    options.update(kwargs)
    return PGSCollection.from_long_weights(weights, **options)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("standardized", [False, True])
def test_matrix_axes_and_scores_match_public_metadata(weights, lazy, standardized):
    original = weights.clone()

    result = from_weights(
        weights.lazy() if lazy else weights,
        standardized=standardized,
        variant_name="rsid",
    )

    assert isinstance(result.weights, csc_array)
    assert result.standardized is standardized
    assert result.genome_build is GenomeBuild.GRCH38
    assert result.features["feature_id"].to_list() == ["gene2", "gene1"]
    assert "pgs_id" not in result.features.columns
    assert result.features["collection_id"].to_list() == ["models", "models"]
    assert result.features["n_variants"].to_list() == [2, 2]
    assert result.variants["variant_id"].to_list() == [
        "1:2:C:CA",
        "1:10:A:G",
        "2:20:C:T",
    ]
    assert result.variants["variant_name"].to_list() == ["rs2", "rs10", "rs20"]
    assert result.variants.schema["chr_position"] == pl.UInt32
    assert_array_equal(
        result.weights.toarray(), [[1.25, 0.0], [0.0, -0.5], [0.75, 0.125]]
    )
    # Rows are samples; genotype columns follow result.variants.
    dosages = np.array([[0, 1, 2], [2, 0, 1]])
    assert_array_equal(dosages @ result.weights, [[1.5, -0.25], [3.25, 0.125]])
    assert result.collections.to_dicts() == [
        {"collection_id": "models", "n_features": 2}
    ]
    for frame in (result.features, result.variants, result.collections):
        assert not {"_variant_idx", "_pgs_idx", "_collection_idx"} & set(frame.columns)
    assert_frame_equal(weights, original)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_supplied_feature_order_and_metadata_determine_matrix_columns(weights, lazy):
    features = pl.DataFrame(
        {
            "feature_id": ["unweighted", "gene1", "gene2"],
            "gene_name": ["THREE", "ONE", "TWO"],
            "pred_r2": [None, 0.2, 0.3],
        }
    )
    original = features.clone()

    with pytest.warns(
        UserWarning, match="removed 1 features with no nonzero weights"
    ) as caught:
        result = from_weights(weights.lazy() if lazy else weights, features=features)

    assert len(caught) == 1
    assert_frame_equal(result.features.select(features.columns), features.slice(1))
    assert_array_equal(
        result.weights.toarray(),
        [[0.0, 1.25], [-0.5, 0.0], [0.125, 0.75]],
    )
    assert result.features["n_variants"].to_list() == [2, 2]
    assert result.collections["n_features"].item() == 2
    assert_frame_equal(features, original)


@pytest.mark.parametrize(
    ("weight_dtype", "metadata_dtype"),
    [
        (pl.Int64, None),
        (pl.Categorical, None),
        (pl.Int64, pl.Int64),
        (pl.Categorical, pl.Categorical),
        (pl.Int64, pl.String),
        (pl.String, pl.Categorical),
    ],
    ids=[
        "integer-inferred",
        "categorical-inferred",
        "integer-metadata",
        "categorical-metadata",
        "integer-weights-string-metadata",
        "string-weights-categorical-metadata",
    ],
)
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_feature_id_normalization_preserves_matrix_alignment(
    weights, weight_dtype, metadata_dtype, lazy
):
    # Use a custom source column and unequal variant counts per feature.
    weights = (
        weights.head(3)
        .rename({"feature_id": "gene"})
        .with_columns(pl.col("gene").str.strip_prefix("gene").cast(weight_dtype))
    )
    original_weights = weights.clone()
    features = None
    if metadata_dtype is not None:
        # Metadata order deliberately differs from first appearance in weights.
        features = pl.DataFrame(
            {"gene": ["1", "2"], "gene_name": ["ONE", "TWO"]}
        ).with_columns(pl.col("gene").cast(metadata_dtype))
    original_features = features.clone() if features is not None else None

    result = from_weights(
        weights.lazy() if lazy else weights, feature_id="gene", features=features
    )

    expected_ids = ["1", "2"] if features is not None else ["2", "1"]
    expected_counts = [1, 2] if features is not None else [2, 1]
    expected_weights = (
        [[0.0, 1.25], [-0.5, 0.0], [0.0, 0.75]]
        if features is not None
        else [[1.25, 0.0], [0.0, -0.5], [0.75, 0.0]]
    )
    assert result.features["feature_id"].dtype == pl.String
    assert result.features["feature_id"].to_list() == expected_ids
    assert result.features["gene"].dtype == pl.String
    assert result.features["gene"].to_list() == expected_ids
    assert result.features["n_variants"].to_list() == expected_counts
    assert_array_equal(result.weights.toarray(), expected_weights)
    assert_frame_equal(weights, original_weights)
    if features is not None and original_features is not None:
        assert result.features["gene_name"].to_list() == ["ONE", "TWO"]
        assert_frame_equal(features, original_features)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_zero_weights_and_unweighted_features_are_removed_with_warnings(
    weights, lazy, with_metadata
):
    weights = weights.with_columns(
        pl.when(pl.col("feature_id") == "gene1")
        .then(0.0)
        .otherwise(pl.col("weight"))
        .alias("weight")
    )
    # gene2 has two nonzero coefficients and one zero coefficient.
    zero_weight = weights.slice(1, 1).with_columns(feature_id=pl.lit("gene2"))
    weights = pl.concat([weights, zero_weight])
    original = weights.clone()
    features = (
        pl.DataFrame({"feature_id": ["gene1", "metadata_only", "gene2"]})
        if with_metadata
        else None
    )

    with pytest.warns(UserWarning) as caught:
        result = from_weights(weights.lazy() if lazy else weights, features=features)

    n_removed_features = 2 if with_metadata else 1
    assert [str(warning.message) for warning in caught] == [
        "Collection 'models': removed 3 zero-weight rows.",
        f"Collection 'models': removed {n_removed_features} features with no nonzero weights.",
    ]
    assert result.features["feature_id"].to_list() == ["gene2"]
    assert result.features["collection_id"].to_list() == ["models"]
    assert result.features["n_variants"].to_list() == [2]
    assert result.collections["n_features"].item() == 1
    assert result.weights.nnz == 2
    assert_array_equal(result.weights.toarray(), [[1.25], [0.0], [0.75]])
    assert_frame_equal(weights, original)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_all_zero_weights_produce_empty_feature_axis(weights, with_metadata):
    weights = weights.with_columns(weight=pl.lit(0.0))
    features = (
        pl.DataFrame({"feature_id": ["gene1", "gene2"]}) if with_metadata else None
    )

    with pytest.warns(UserWarning) as caught:
        result = from_weights(weights, features=features)

    assert [str(warning.message) for warning in caught] == [
        "Collection 'models': removed 4 zero-weight rows.",
        "Collection 'models': removed 2 features with no nonzero weights.",
    ]
    assert result.features.is_empty()
    assert result.features["n_variants"].to_list() == []
    assert result.collections["n_features"].item() == 0
    assert result.weights.shape == (result.variants.height, 0)
    assert result.weights.nnz == 0


def test_n_variants_replaces_stale_counts_and_preserves_reported_counts(weights):
    # Different counts and reversed metadata order expose misaligned annotations.
    weights = weights.head(3).rename({"feature_id": "gene"})
    features = pl.DataFrame(
        {
            "gene": ["gene1", "gene2"],
            "n_variants": [99, 98],
            "n_variants_reported": [3, 4],
        }
    )
    original = features.clone()

    result = from_weights(weights, feature_id="gene", features=features)

    assert result.features["feature_id"].to_list() == ["gene1", "gene2"]
    assert result.features["n_variants"].to_list() == [1, 2]
    assert result.features["n_variants_reported"].to_list() == [3, 4]
    assert "n_variants_right" not in result.features.columns
    assert_array_equal(
        result.weights.toarray(), [[0.0, 1.25], [-0.5, 0.0], [0.0, 0.75]]
    )
    assert_frame_equal(features, original)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_custom_columns_preserve_original_variant_ids_as_names(weights, lazy):
    weights = weights.rename(
        {
            "feature_id": "gene",
            "chr_name": "chrom",
            "chr_position": "pos",
            "other_allele": "ref",
            "effect_allele": "alt",
            "weight": "beta",
            "rsid": "variant_id",
        }
    )
    original = weights.clone()

    result = from_weights(
        weights.lazy() if lazy else weights,
        feature_id="gene",
        effect_weight="beta",
        chr_name="chrom",
        chr_position="pos",
        other_allele="ref",
        effect_allele="alt",
        variant_name="variant_id",
    )

    assert result.features["gene"].to_list() == ["gene2", "gene1"]
    assert result.features["feature_id"].to_list() == ["gene2", "gene1"]
    assert result.variants["variant_name"].to_list() == ["rs2", "rs10", "rs20"]
    assert result.variants["variant_id"].to_list() == [
        "1:2:C:CA",
        "1:10:A:G",
        "2:20:C:T",
    ]
    assert_array_equal(
        result.weights.toarray(), [[1.25, 0.0], [0.0, -0.5], [0.75, 0.125]]
    )
    assert_frame_equal(weights, original)


@pytest.mark.parametrize(
    ("renames", "options"),
    [
        ({"feature_id": "variant_id"}, {"feature_id": "variant_id"}),
        ({"weight": "variant_id"}, {"effect_weight": "variant_id"}),
        (
            {"feature_id": "chr_name", "chr_name": "chrom"},
            {"feature_id": "chr_name", "chr_name": "chrom"},
        ),
        (
            {"weight": "feature_id", "feature_id": "gene"},
            {"effect_weight": "feature_id", "feature_id": "gene"},
        ),
    ],
)
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("supplied_metadata", [False, True])
def test_source_names_overlap_canonical_columns(
    weights, renames, options, lazy, supplied_metadata
):
    weights = weights.rename(renames)
    feature_column = options.get("feature_id", "feature_id")
    features = (
        weights.select(feature_column).unique(maintain_order=True).reverse()
        if supplied_metadata
        else None
    )

    result = from_weights(
        weights.lazy() if lazy else weights,
        features=features,
        variant_name="rsid",
        **options,
    )

    expected_ids = ["gene1", "gene2"] if supplied_metadata else ["gene2", "gene1"]
    expected_weights = (
        [[0.0, 1.25], [-0.5, 0.0], [0.125, 0.75]]
        if supplied_metadata
        else [[1.25, 0.0], [0.0, -0.5], [0.75, 0.125]]
    )
    assert result.features["feature_id"].to_list() == expected_ids
    assert result.variants["variant_name"].to_list() == ["rs2", "rs10", "rs20"]
    assert_array_equal(result.weights.toarray(), expected_weights)


@pytest.mark.parametrize("column", ["feature_id", "gene", "pgs_id", "collection_id"])
@pytest.mark.parametrize("context", [None, "blood"])
@pytest.mark.parametrize("supplied_metadata", [False, True])
def test_feature_ids_survive_metadata_construction(column, context, supplied_metadata):
    weights = pl.DataFrame(
        {
            column: ["G1", "G1_blood"],
            "chr_name": ["1", "1"],
            "chr_position": [10, 20],
            "other_allele": ["A", "C"],
            "effect_allele": ["G", "T"],
            "weight": [0.25, -0.5],
        }
    )
    features = pl.DataFrame({column: ["G1_blood", "G1"]}) if supplied_metadata else None

    result = from_weights(
        weights, feature_id=column, features=features, context=context
    )

    if supplied_metadata:
        expected_ids = ["G1_blood", "G1"]
        expected_weights = [[0.0, 0.25], [-0.5, 0.0]]
    else:
        expected_ids = ["G1", "G1_blood"]
        expected_weights = [[0.25, 0.0], [0.0, -0.5]]
    assert result.features["feature_id"].to_list() == expected_ids
    if column == "pgs_id":
        # An explicitly mapped source column is preserved as ordinary metadata.
        assert result.features[column].to_list() == expected_ids
    else:
        assert "pgs_id" not in result.features.columns
    assert result.features["collection_id"].to_list() == ["models", "models"]
    assert_array_equal(result.weights.toarray(), expected_weights)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"context": "blood"},
        {
            "method": "enet",
            "context": "blood",
            "dataset": "GTEx",
            "modality": "expression",
        },
    ],
)
def test_collection_metadata_preserves_score_identity(weights, metadata):
    result = from_weights(weights, **metadata)

    assert result.collections.to_dicts() == [
        {"collection_id": "models", **metadata, "n_features": 2}
    ]
    assert result.features.select("collection_id", "feature_id").rows() == [
        ("models", "gene2"),
        ("models", "gene1"),
    ]
    assert "pgs_id" not in result.features.columns
    for column, value in metadata.items():
        assert result.features[column].to_list() == [value, value]


def test_collection_id_distinguishes_scores_for_the_same_features(weights):
    first = from_weights(weights, collection_id="first", context="blood")
    second = from_weights(weights, collection_id="second", context="blood")

    first_ids = first.features.select("collection_id", "feature_id")
    second_ids = second.features.select("collection_id", "feature_id")
    assert first_ids.rows() == [("first", "gene2"), ("first", "gene1")]
    assert second_ids.rows() == [("second", "gene2"), ("second", "gene1")]
    assert not pl.concat([first_ids, second_ids]).is_duplicated().any()
    assert first.collections["collection_id"].item() == "first"
    assert second.collections["collection_id"].item() == "second"
    assert_array_equal(first.weights.toarray(), second.weights.toarray())


def test_rejects_null_collection_id(weights):
    with pytest.raises(ValueError, match="collection_id is required"):
        from_weights(weights, collection_id=None)


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_resolves_collection_metadata_from_features(weights, explicit_metadata):
    metadata = {
        "dataset": "GTEx",
        "context": "blood",
        "modality": "expression",
        "method": "enet",
    }
    features = pl.DataFrame(
        {
            "feature_id": ["gene1", "gene2"],
            "dataset": ["GTEx", "GTEx"],
            "context": ["blood", "blood"],
            "modality": ["expression", "expression"],
            "method": ["enet", "enet"],
        }
    )
    original = features.clone()

    result = from_weights(
        weights, features=features, **(metadata if explicit_metadata else {})
    )

    assert result.collections.to_dicts() == [
        {"collection_id": "models", **metadata, "n_features": 2}
    ]
    assert result.features.select("collection_id", "feature_id").rows() == [
        ("models", "gene1"),
        ("models", "gene2"),
    ]
    assert "pgs_id" not in result.features.columns
    for column, value in metadata.items():
        assert result.features[column].to_list() == [value, value]
    assert_array_equal(
        result.weights.toarray(), [[0.0, 1.25], [-0.5, 0.0], [0.125, 0.75]]
    )
    assert_frame_equal(features, original)

    restored = from_weights(weights, features=result.features)
    assert_frame_equal(restored.features, result.features)
    assert_frame_equal(restored.collections, result.collections)
    assert_array_equal(restored.weights.toarray(), result.weights.toarray())


@pytest.mark.parametrize("column", ["dataset", "context", "modality", "method"])
@pytest.mark.parametrize("values", [["source", None], [None, None]])
@pytest.mark.parametrize("argument", [None, "source"])
def test_rejects_null_collection_metadata(weights, column, values, argument):
    features = pl.DataFrame({"feature_id": ["gene1", "gene2"], column: values})

    with pytest.raises(ValueError, match=rf"{column!r}.*null"):
        from_weights(weights, features=features, **{column: argument})


@pytest.mark.parametrize("context", [None, "blood"])
def test_rejects_empty_collection_metadata_column(weights, context):
    features = pl.DataFrame(schema={"feature_id": pl.String, "context": pl.String})

    with pytest.raises(ValueError, match="'context'.*exactly one"):
        from_weights(weights.head(0), features=features, context=context)


@pytest.mark.parametrize("column", ["dataset", "context", "modality", "method"])
def test_rejects_conflicting_collection_metadata(weights, column):
    features = pl.DataFrame(
        {"feature_id": ["gene1", "gene2"], column: ["source", "source"]}
    )

    with pytest.raises(ValueError, match=rf"{column}=.*conflicts"):
        from_weights(weights, features=features, **{column: "different"})


@pytest.mark.parametrize("column", ["dataset", "context", "modality", "method"])
@pytest.mark.parametrize("argument", [None, "first"])
def test_rejects_multiple_collection_values_with_unique_feature_ids(
    weights, column, argument
):
    features = pl.DataFrame(
        {"feature_id": ["gene1", "gene2"], column: ["first", "second"]}
    )

    with pytest.raises(ValueError, match=rf"{column!r}.*exactly one"):
        from_weights(weights, features=features, **{column: argument})


def test_feature_id_source_column_is_not_collection_metadata(weights):
    weights = weights.rename({"feature_id": "context"})
    features = pl.DataFrame({"context": ["gene1", "gene2"]})

    result = from_weights(
        weights, feature_id="context", features=features, context="blood"
    )

    assert result.features["feature_id"].to_list() == ["gene1", "gene2"]
    assert result.features["context"].to_list() == ["blood", "blood"]
    assert result.collections["context"].item() == "blood"
    assert "pgs_id" not in result.features.columns
    assert_array_equal(
        result.weights.toarray(), [[0.0, 1.25], [-0.5, 0.0], [0.125, 0.75]]
    )


def test_variant_names_are_optional(weights):
    result = from_weights(weights.drop("rsid"))

    assert "variant_name" not in result.variants.columns
    assert result.weights.shape == (3, 2)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_empty_weights_drop_unweighted_features(weights, with_metadata):
    features = pl.DataFrame({"feature_id": ["gene1"]}) if with_metadata else None

    if with_metadata:
        with pytest.warns(
            UserWarning, match="removed 1 features with no nonzero weights"
        ) as caught:
            result = from_weights(weights.head(0), features=features)
        assert len(caught) == 1
    else:
        result = from_weights(weights.head(0))

    assert result.weights.shape == (0, 0)
    assert result.weights.nnz == 0
    assert result.features.is_empty()
    assert result.variants.schema["chr_position"] == pl.UInt32
    assert result.collections["n_features"].item() == 0


@pytest.mark.parametrize(
    "column",
    [
        "feature_id",
        "chr_name",
        "chr_position",
        "other_allele",
        "effect_allele",
        "weight",
    ],
)
def test_rejects_missing_required_weight_columns(weights, column):
    with pytest.raises(ValueError, match="not in weights"):
        from_weights(weights.drop(column))


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_rejects_missing_requested_variant_names(weights, lazy):
    weights = weights.drop("rsid")

    with pytest.raises(ValueError, match="Columns .*rsid.* not in weights"):
        from_weights(weights.lazy() if lazy else weights, variant_name="rsid")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_rejects_invalid_weights(weights, value, lazy):
    weights = weights.with_columns(pl.lit(value, dtype=pl.Float64).alias("weight"))

    with pytest.raises(ValueError, match="weights"):
        from_weights(weights.lazy() if lazy else weights)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_rejects_duplicate_feature_variant_pairs_after_normalization(weights, lazy):
    duplicate = weights.head(1).with_columns(
        chr_name=pl.lit("2"), chr_position=pl.lit("020")
    )
    weights = pl.concat([weights, duplicate])

    with pytest.raises(ValueError, match="Duplicate feature x variant"):
        from_weights(weights.lazy() if lazy else weights)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_rejects_null_feature_ids_in_weights(weights, with_metadata):
    features = weights.select("feature_id").unique() if with_metadata else None
    weights = weights.with_columns(pl.lit(None, dtype=pl.String).alias("feature_id"))

    with pytest.raises(ValueError, match="Missing feature ids in weights"):
        from_weights(weights.head(1), features=features)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (pl.DataFrame({"gene": ["gene1", "gene2"]}), "must be present in features"),
        (
            pl.DataFrame({"feature_id": ["gene1", None]}),
            "Missing feature ids in features",
        ),
        (
            pl.DataFrame({"feature_id": ["gene1", "gene1"]}),
            "Duplicate feature ids in features",
        ),
        (pl.DataFrame({"feature_id": ["gene1"]}), "missing from features metadata"),
    ],
)
def test_rejects_invalid_feature_metadata(weights, features, message):
    with pytest.raises(ValueError, match=message):
        from_weights(weights, features=features)


@pytest.mark.parametrize("column", ["other_allele", "effect_allele"])
@pytest.mark.parametrize("value", [None, ""])
def test_rejects_missing_alleles(weights, column, value):
    weights = weights.head(1).with_columns(pl.lit(value, dtype=pl.String).alias(column))

    with pytest.raises(ValueError, match=column):
        from_weights(weights)


@pytest.mark.parametrize("build", [None, "unknown"])
def test_rejects_missing_or_unknown_genome_build(weights, build):
    with pytest.raises(
        ValueError, match="genome_build is required|Unknown genome build"
    ):
        from_weights(weights, genome_build=build)


@pytest.fixture
def predictdb(tmp_path):
    path = tmp_path / "synthetic_model.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            'CREATE TABLE extra (gene TEXT, genename TEXT, "n.snps.in.model" INTEGER, '
            '"pred.perf.R2" REAL, "pred.perf.pval" REAL, annotation TEXT)'
        )
        connection.executemany(
            "INSERT INTO extra VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("gene1", "ONE", 2, 0.2, 0.01, "first"),
                ("gene2", "TWO", 2, 0.3, 0.02, "second"),
            ],
        )
        connection.execute(
            "CREATE TABLE weights (rsid TEXT, gene TEXT, weight REAL, "
            "ref_allele TEXT, eff_allele TEXT, varID TEXT)"
        )
        connection.executemany(
            "INSERT INTO weights VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("rs20", "gene2", 0.75, "C", "T", "chr2_20_C_T_b38"),
                # Explicit allele columns, not varID, define effect orientation.
                ("rs10", "gene1", -0.5, "A", "G", "chr1_10_G_A_b38"),
                ("rs2", "gene2", 1.25, "CA", "C", "chr1_2_CA_C_b38"),
                ("rs20", "gene1", 0.125, "C", "T", "chr2_20_C_T_b38"),
            ],
        )
    return path


@pytest.mark.parametrize(
    ("genome_build", "collection_id"),
    [(None, None), ("hg38", "named_model"), (GenomeBuild.GRCH38, None)],
)
def test_predictdb_uses_explicit_alleles_and_preserves_metadata(
    predictdb, genome_build, collection_id
):
    result = PGSCollection.from_predictdb(
        predictdb,
        genome_build=genome_build,
        collection_id=collection_id,
        context="blood",
    )

    expected_id = collection_id or "synthetic_model"
    assert result.genome_build is GenomeBuild.GRCH38
    assert result.standardized is False
    assert result.features["feature_id"].to_list() == ["gene1", "gene2"]
    assert result.features["gene_id"].to_list() == ["gene1", "gene2"]
    assert result.features["gene_name"].to_list() == ["ONE", "TWO"]
    assert result.features["pred_r2"].to_list() == [0.2, 0.3]
    assert result.features["pred_pval"].to_list() == [0.01, 0.02]
    assert result.features["n_variants_reported"].to_list() == [2, 2]
    assert result.features["n_variants"].to_list() == [2, 2]
    assert result.features["annotation"].to_list() == ["first", "second"]
    assert result.features["collection_id"].to_list() == [expected_id, expected_id]
    assert "pgs_id" not in result.features.columns
    assert result.variants["variant_id"].to_list() == [
        "1:2:CA:C",
        "1:10:A:G",
        "2:20:C:T",
    ]
    assert result.variants["variant_name"].to_list() == ["rs2", "rs10", "rs20"]
    assert_array_equal(
        result.weights.toarray(), [[0.0, 1.25], [-0.5, 0.0], [0.125, 0.75]]
    )
    assert result.collections.to_dicts() == [
        {"collection_id": expected_id, "context": "blood", "n_features": 2}
    ]


def test_predictdb_infers_grch37(predictdb):
    with sqlite3.connect(predictdb) as connection:
        connection.execute("UPDATE weights SET varID = replace(varID, '_b38', '_b37')")

    assert PGSCollection.from_predictdb(predictdb).genome_build is GenomeBuild.GRCH37


def test_predictdb_rejects_mismatched_genome_build(predictdb):
    with pytest.raises(ValueError, match="Genome build mismatch"):
        PGSCollection.from_predictdb(predictdb, genome_build="hg19")


def test_predictdb_rejects_mixed_genome_builds(predictdb):
    with sqlite3.connect(predictdb) as connection:
        connection.execute(
            "UPDATE weights SET varID = replace(varID, '_b38', '_b37') WHERE rsid = 'rs10'"
        )

    with pytest.raises(ValueError, match="Found multiple genome builds"):
        PGSCollection.from_predictdb(predictdb)


def test_predictdb_rejects_missing_database_without_creating_it(tmp_path):
    path = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError, match="PredictDB database not found"):
        PGSCollection.from_predictdb(path)
    assert not path.exists()
