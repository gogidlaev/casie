import pytest

from casie.build import GenomeBuild


def test_canonical_builds():
    assert GenomeBuild.GRCH38.value == "GRCh38"
    assert GenomeBuild.GRCH37.value == "GRCh37"

@pytest.mark.parametrize(
    "input_str,expected_build",
    [
        ("GRCh38", GenomeBuild.GRCH38),
        ("grch38", GenomeBuild.GRCH38),
        ("Hg38", GenomeBuild.GRCH38),
        ("hg38", GenomeBuild.GRCH38),
        ("b38", GenomeBuild.GRCH38),
        ("GRCh37", GenomeBuild.GRCH37),
        ("grch37", GenomeBuild.GRCH37),
        ("Hg19", GenomeBuild.GRCH37),
        ("hg19", GenomeBuild.GRCH37),
        ("b37", GenomeBuild.GRCH37)
    ]
)
def test_parse_from_valid_input_str(input_str, expected_build):
    assert GenomeBuild.parse_from(input_str) is expected_build

@pytest.mark.parametrize(
    "build",
    [
        GenomeBuild.GRCH38,
        GenomeBuild.GRCH37
    ]
)
def test_parse_from_instance(build):
    assert GenomeBuild.parse_from(build) is build

@pytest.mark.parametrize(
    "input_str",
    [
        "",
        "unknown_build",
        "GRCh39",
        "hg20",
        "39"
    ]
)
def test_parse_from_invalid_input_str(input_str):
    with pytest.raises(ValueError, match=f"Unknown genome build: {input_str}"):
        GenomeBuild.parse_from(input_str)

@pytest.mark.parametrize(
    "input_value",
    [
        39,
        123,
        3.14,
        None,
        [],
        {},
        object()
    ]
)
def test_parse_from_invalid_input_type(input_value):
    with pytest.raises(TypeError, match=f"GenomeBuild can only be parsed from string, not {type(input_value)}"):
        GenomeBuild.parse_from(input_value)