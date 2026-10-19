import polars as pl
import pytest
from polars.testing import assert_frame_equal

from casie.build import GenomeBuild
from casie.variants import VariantFrame


@pytest.fixture
def frame():
    return pl.DataFrame(
        {
            "chr_name": ["2", "1"],
            "chr_position": pl.Series([20, 2], dtype=pl.UInt32),
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "variant_name": ["rs20", "rs2"],
            "_variant_idx": [7, 3],
            "annotation": ["first", "second"],
        }
    )


def from_frame(frame, **kwargs):
    return VariantFrame(frame, a1="ref", a2="alt", genome_build="hg38", **kwargs)


def test_constructs_ids_without_reordering_or_mutating_input(frame):
    original = frame.clone()

    result = from_frame(frame)

    expected = frame.with_columns(
        pl.Series("variant_id", ["2:20:A:G", "1:2:C:T"]),
    )
    assert_frame_equal(result.frame, expected, check_column_order=False)
    assert_frame_equal(frame, original)
    assert result.genome_build is GenomeBuild.GRCH38
    assert (result.a1, result.a2) == ("ref", "alt")


def test_accepts_matching_existing_ids(frame):
    expected = from_frame(frame).frame

    assert_frame_equal(from_frame(expected).frame, expected)


def test_rejects_mismatched_existing_ids(frame):
    frame = frame.with_columns(pl.Series("variant_id", ["2:20:G:A", "1:2:C:T"]))

    with pytest.raises(ValueError, match="Existing variant_id does not match"):
        from_frame(frame)


@pytest.mark.parametrize("column", ["chr_name", "chr_position", "ref", "alt"])
def test_rejects_missing_required_columns(frame, column):
    with pytest.raises(ValueError, match="not in variants"):
        from_frame(frame.drop(column))


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("chr_name", "Missing chromosome names"),
        ("chr_position", "Missing chromosome positions"),
        ("ref", "contains null values"),
        ("alt", "contains null values"),
    ],
)
def test_rejects_null_variant_fields(frame, column, message):
    frame = frame.with_columns(pl.lit(None, dtype=frame.schema[column]).alias(column))

    with pytest.raises(ValueError, match=message):
        from_frame(frame)


@pytest.mark.parametrize("column", ["chr_name", "ref", "alt"])
def test_rejects_empty_variant_fields(frame, column):
    frame = frame.with_columns(pl.lit("").alias(column))

    with pytest.raises(ValueError, match="[Ee]mpty"):
        from_frame(frame)


@pytest.mark.parametrize(("ref", "alt"), [("C", "CA"), ("CA", "C"), ("I", "D")])
def test_accepts_nonempty_indel_alleles(frame, ref, alt):
    frame = frame.head(1).with_columns(ref=pl.lit(ref), alt=pl.lit(alt))

    result = from_frame(frame)

    assert result.frame["variant_id"].item() == f"2:20:{ref}:{alt}"


@pytest.mark.parametrize("existing_id", [False, True])
def test_rejects_duplicate_ids_even_with_different_metadata(frame, existing_id):
    if existing_id:
        frame = from_frame(frame).frame
    duplicate = frame.head(1).with_columns(annotation=pl.lit("different"))

    with pytest.raises(ValueError, match="Duplicate variant ids"):
        from_frame(pl.concat([frame, duplicate]))


@pytest.mark.parametrize("dtype", [pl.String, pl.Int64, pl.UInt64, pl.Float64])
def test_rejects_position_dtypes_other_than_uint32(frame, dtype):
    frame = frame.with_columns(pl.col("chr_position").cast(dtype))

    with pytest.raises(
        ValueError, match=r"Chromosome positions must have pl\.UInt32 dtype"
    ):
        from_frame(frame)


@pytest.mark.parametrize("existing_id", [False, True])
def test_preserves_metadata_named_temp_variant_id(frame, existing_id):
    if existing_id:
        frame = from_frame(frame).frame
    frame = frame.with_columns(temp_variant_id=pl.lit("keep me"))

    assert from_frame(frame).frame["temp_variant_id"].to_list() == ["keep me"] * 2


def test_variant_name_is_optional(frame):
    result = from_frame(frame.drop("variant_name"))

    assert "variant_name" not in result.frame.columns
    assert result.frame["variant_id"].to_list() == ["2:20:A:G", "1:2:C:T"]


def test_preserves_hidden_indices_and_records_hidden_columns(frame):
    result = from_frame(frame, hide_cols=["_variant_idx"])

    assert result.hide_cols == ["_variant_idx"]
    assert result.frame.columns[0] == "_variant_idx"
    assert result.frame["_variant_idx"].to_list() == [7, 3]


@pytest.mark.parametrize("column", ["ref", "variant_id", "variant_name"])
def test_rejects_hiding_variant_metadata(frame, column):
    with pytest.raises(ValueError, match="Hidden columns cannot"):
        from_frame(frame, hide_cols=[column])


def test_accepts_empty_frame_with_defined_schema(frame):
    result = from_frame(frame.head(0)).frame

    assert result.height == 0
    assert result.schema["variant_id"] == pl.String
    assert result.schema["chr_position"] == pl.UInt32


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("column", ["chr_position", "pos"])
@pytest.mark.parametrize("dtype", [pl.String, pl.Int64])
def test_normalizes_positions_without_changing_other_data(lazy, column, dtype):
    frame = pl.DataFrame(
        {
            column: pl.Series([20, 2, None], dtype=pl.Int64).cast(dtype),
            "annotation": ["first", "second", "third"],
        }
    )
    original = frame.clone()
    expected = frame.with_columns(pl.Series(column, [20, 2, None], dtype=pl.UInt32))
    kwargs = {} if column == "chr_position" else {"chr_position": column}

    result = VariantFrame.normalize_position(frame.lazy() if lazy else frame, **kwargs)

    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert_frame_equal(result, expected)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("position", ["-1", "not-a-position", "4294967296"])
def test_normalize_position_rejects_invalid_casts(position, lazy):
    frame = pl.DataFrame({"chr_position": [position]})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        result = VariantFrame.normalize_position(frame.lazy() if lazy else frame)
        if isinstance(result, pl.LazyFrame):
            result.collect()


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_normalizes_aliases_and_preserves_unknown_contigs(lazy):
    frame = pl.DataFrame(
        {
            "chrom": [
                " chr1 ",
                "CHR22",
                "x",
                "chrY",
                "M",
                "chrM",
                "MT",
                "chrUn_A",
                None,
            ],
            "value": list(range(9)),
        }
    )
    expected = frame.with_columns(
        pl.Series("chrom", ["1", "22", "X", "Y", "MT", "MT", "MT", "chrUn_A", None])
    )

    result = VariantFrame.normalize_chr(
        frame.lazy() if lazy else frame, chr_name="chrom"
    )

    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_adds_id_with_custom_column_names_and_alias(frame, lazy):
    frame = frame.rename({"chr_name": "chrom", "chr_position": "pos"})
    expected = frame.with_columns(pl.Series("key", ["2:20:A:G", "1:2:C:T"]))

    result = VariantFrame.add_variant_id(
        frame.lazy() if lazy else frame,
        chr_name="chrom",
        chr_position="pos",
        a1="ref",
        a2="alt",
        alias="key",
    )

    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_sorts_chromosomes_contigs_positions_and_alleles(lazy):
    expected = pl.DataFrame(
        [
            ("1", 2, "A", "C"),
            ("1", 2, "A", "G"),
            ("1", 2, "C", "A"),
            ("1", 10, "A", "G"),
            ("2", 1, "A", "G"),
            ("10", 1, "A", "G"),
            ("X", 1, "A", "G"),
            ("Y", 1, "A", "G"),
            ("MT", 1, "A", "G"),
            ("contig_A", 2, "A", "G"),
            ("contig_A", 10, "A", "G"),
            ("contig_B", 1, "A", "G"),
        ],
        schema=["chr_name", "chr_position", "other_allele", "effect_allele"],
        orient="row",
    ).with_row_index("_variant_idx")
    frame = expected.reverse()

    result = VariantFrame.sort_variants(frame.lazy() if lazy else frame)

    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    assert_frame_equal(result, expected)
