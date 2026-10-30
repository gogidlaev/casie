from __future__ import annotations

import polars as pl

from .build import GenomeBuild


class VariantFrame:
    """
    Internal format for harmonized processing and representation of variant metadata.
    """

    _canonical_chr = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

    def __init__(
        self,
        variants: pl.DataFrame,
        a1: str,
        a2: str,
        genome_build: str | GenomeBuild,
        hide_cols: list[str] | None = None,
    ):
        self.genome_build = GenomeBuild.parse_from(genome_build)

        cols = ["chr_name", "chr_position", a1, a2]
        missing_cols = set(cols).difference(set(variants.columns))
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not in variants")

        # Validate chr_names and chr_position
        chr_names = variants.get_column("chr_name")
        if chr_names.null_count():
            raise ValueError("Missing chromosome names")
        if chr_names.eq("").any():
            raise ValueError("Empty chromosome names")

        chr_positions = variants.get_column("chr_position")
        if chr_positions.null_count():
            raise ValueError("Missing chromosome positions")
        if chr_positions.dtype != pl.UInt32:
            raise ValueError("Chromosome positions must have pl.UInt32 dtype")

        for col in (a1, a2):
            alleles = variants.get_column(col)
            if alleles.null_count():
                raise ValueError(f"{col} contains null values")
            if alleles.eq("").any():
                raise ValueError(f"{col} contains empty values")

        expected_id = variants.select(
            pl.concat_str(["chr_name", "chr_position", a1, a2], separator=":").alias(
                "variant_id"
            )
        ).to_series()

        # We need to enforce the correct variant_id
        if "variant_id" in variants.columns:
            if not variants.get_column("variant_id").equals(expected_id):
                raise ValueError(
                    "Existing variant_id does not match the corresponding variant fields. Ensure that variant_id is constructed after chromosome normalization"
                )
        else:
            variants = variants.with_columns(expected_id)

        variant_ids = variants.get_column("variant_id")
        if variant_ids.null_count():
            raise ValueError("Missing variant ids are not allowed")
        if variant_ids.is_duplicated().any():
            raise ValueError("Duplicate variant ids are not allowed")

        cols.append("variant_id")

        if "variant_name" in variants.columns:
            cols.append("variant_name")

        if hide_cols is not None:
            if any(c in cols for c in hide_cols):
                raise ValueError(f"New columns cannot be one of {cols}")

            cols = hide_cols + cols

        other_cols = [c for c in variants.columns if c not in cols]
        if other_cols:
            cols += other_cols

        self.frame = variants.select(cols)
        self.hide_cols = hide_cols

        self.a1 = a1
        self.a2 = a2

    @staticmethod
    def normalize_chr(frame: pl.DataFrame | pl.LazyFrame, chr_name: str = "chr_name"):
        chr_aliases = {name: name for name in VariantFrame._canonical_chr} | {"M": "MT"}

        raw_chr = pl.col(chr_name).cast(pl.String).str.strip_chars()
        candidate_chr = raw_chr.str.replace(r"(?i)^chr", "").str.to_uppercase()

        return frame.with_columns(
            candidate_chr.replace_strict(chr_aliases, default=raw_chr)
        )

    @staticmethod
    def normalize_position(
        frame: pl.DataFrame | pl.LazyFrame, chr_position: str = "chr_position"
    ):
        return frame.with_columns(pl.col(chr_position).cast(pl.UInt32, strict=True))

    @staticmethod
    def add_variant_id(
        frame: pl.DataFrame | pl.LazyFrame,
        chr_name: str = "chr_name",
        chr_position: str = "chr_position",
        a1: str = "other_allele",
        a2: str = "effect_allele",
        alias: str = "variant_id",
    ):
        return frame.with_columns(
            pl.concat_str([chr_name, chr_position, a1, a2], separator=":").alias(alias)
        )

    @staticmethod
    def sort_variants(
        frame: pl.DataFrame | pl.LazyFrame,
        chr_name: str = "chr_name",
        chr_position: str = "chr_position",
        a1: str = "other_allele",
        a2: str = "effect_allele",
    ):
        chr_ranks = {c: i for i, c in enumerate(VariantFrame._canonical_chr)}

        return frame.sort(
            [
                pl.col(chr_name).replace_strict(
                    chr_ranks, default=len(VariantFrame._canonical_chr)
                ),
                pl.col(chr_name),
                pl.col(chr_position),
                pl.col(a1),
                pl.col(a2),
            ]
        )
