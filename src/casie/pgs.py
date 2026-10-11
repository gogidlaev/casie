from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import polars as pl
from scipy.sparse import coo_array, csc_array

from .build import GenomeBuild


@dataclass(frozen=True, slots=True, repr=False)
class PGSCollection:
    weights: csc_array
    _features: pl.DataFrame
    _variants: pl.DataFrame
    genome_build: GenomeBuild

    @classmethod
    def from_predictdb(
        cls,
        db: str | Path,
        genome_build: str | GenomeBuild | None = None
    ) -> PGSCollection:
        import sqlite3

        db_path = Path(db)
        if not db_path.exists():
            raise FileNotFoundError(f"PredictDB database not found: {db_path}")

        with sqlite3.connect(db_path) as conn:
            extra = (
                pl.read_database("SELECT * FROM extra", conn)
                .select(
                    pl.col("gene").alias("feature_id"),
                    pl.col("gene").alias("gene_id"),
                    pl.col("genename").alias("gene_name"),
                    pl.col("n.snps.in.model").alias("n_variants_reported"),
                    pl.col("pred.perf.R2").alias("pred_r2"),
                    pl.col("pred.perf.pval").alias("pred_pval")
                )
                .with_row_index("_feature_idx")
            )

            weights = (
                pl.read_database("SELECT * FROM weights", conn)
                .select(
                    pl.col("gene").alias("feature_id"),
                    pl.col("varID").str.split("_").list.to_struct(
                        fields=["chr_name", "chr_position", "other_allele", "effect_allele", "genome_build"]
                    ).struct.unnest(),
                    pl.col("rsid").alias("variant_name"),
                    pl.col("weight")
                )
                .with_columns(
                    pl.col("chr_name").str.to_lowercase().str.strip_prefix("chr").cast(pl.UInt8),
                    pl.col("chr_position").cast(pl.UInt32),
                )
            )

            genome_build_inferred = weights["genome_build"].unique()
            if len(genome_build_inferred) > 1:
                raise ValueError(f"Found multiple genome builds in the weights table: {genome_build_inferred}")
            genome_build_inferred = GenomeBuild.parse_from(genome_build_inferred.cast(pl.String)[0])
            
            if genome_build is None:
                genome_build = genome_build_inferred
            else:
                genome_build = GenomeBuild.parse_from(genome_build)
                if genome_build != genome_build_inferred:
                    raise ValueError(
                        f"Genome build mismatch: provided {genome_build.value}, "
                        f"but found {genome_build_inferred.value} in the weights table"
                    )

            weights = weights.with_columns(
                pl.concat_str(["chr_name", "chr_position", "other_allele", "effect_allele"], separator=":").alias("variant_id")
            )

            variants = (
                weights
                .select(["chr_name", "chr_position", "other_allele", "effect_allele", "variant_id", "variant_name"])
                .unique("variant_id")
                .sort(["chr_name", "chr_position"])
                .with_row_index("_variant_idx")
            )
            
            weights = (
                weights
                .select(["feature_id", "variant_id", "weight"])
                .join(variants.select(["variant_id", "_variant_idx"]), on="variant_id")
                .join(extra.select(["feature_id", "_feature_idx"]), on="feature_id")
                .select(["_variant_idx", "_feature_idx", "weight"])
            )

            weights = coo_array(
                (weights["weight"].to_numpy(), (weights["_variant_idx"].to_numpy(), weights["_feature_idx"].to_numpy())),
                shape=(variants.height, extra.height)
            ).tocsc()

            return cls(weights, extra, variants, genome_build)

    @property
    def features(self):
        return self._features.drop("_feature_idx")

    @property
    def variants(self):
        return self._variants.drop("_variant_idx")
        