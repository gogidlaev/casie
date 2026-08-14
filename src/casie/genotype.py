from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from pgenlib import PgenReader, PvarReader

from .build import GenomeBuild


class GenotypeFile(ABC):
    @abstractmethod
    def read_dosages(
        self, 
        variant_idx: pl.Series, 
        obs_idx: pl.Series
    ) -> np.ndarray: 
        ...


class PgenFile(GenotypeFile):
    def __init__(
        self, 
        pgen: Path,
        pvar: Path
    ):
        self.pgen = pgen
        self.pvar = pvar

    def read_dosages(
        self, 
        variant_idx: pl.Series, 
        obs_idx: pl.Series
    ) -> np.ndarray:
        with PvarReader(bytes(self.pvar)) as pvar_reader:
            with PgenReader(
                bytes(self.pgen), 
                sample_subset=obs_idx.to_numpy(), 
                pvar=pvar_reader
            ) as pgen_reader:
                dosages = np.empty(
                    shape=(variant_idx.len(), obs_idx.len()), 
                    dtype=np.float32
                )
                pgen_reader.read_dosages_list(variant_idx.to_numpy(), dosages)

                return dosages


@dataclass(frozen=True, slots=True, repr=False)
class GenotypeDataset:
    gen_file: GenotypeFile
    _obs: pl.DataFrame
    _variants: pl.LazyFrame
    genome_build: GenomeBuild

    @classmethod
    def from_pgen(
        cls, 
        pgen: str | Path,
        genome_build: str | GenomeBuild
    ):
        genome_build = GenomeBuild.parse_from(genome_build)

        pgen_file = Path(pgen)
        psam_file = pgen_file.with_suffix(".psam")
        pvar_file = pgen_file.with_suffix(".pvar.zst")

        gen_file = PgenFile(pgen_file, pvar_file)

        obs = pl.read_csv(
            psam_file, 
            separator="\t",
            infer_schema_length=10000,
            null_values=["NA", "nan"]
        ).with_row_index("_obs_idx")

        canonical_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]

        variants = (
            pl.scan_csv(
                pvar_file,
                separator="\t",
                comment_prefix="##"
            )
            .with_row_index("_variant_idx")
            .select(
                pl.col("#CHROM").alias("chr_name"),
                pl.col("POS").alias("chr_position"),
                pl.col("ID").alias("variant_name"),
                pl.col("REF").alias("ref_allele"),
                pl.col("ALT").alias("alt_allele")
            )
            .with_columns(
                pl.col("chr_name").str.to_lowercase().str.strip_prefix(r"(?i)^chr"),
                pl.col("chr_position").cast(pl.UInt32),
                pl.concat_str(pl.col(["chr_name", "chr_position", "ref_allele", "alt_allele"]), separator="_").alias("variant_id")
            )
            .filter(pl.col("chr_name").is_in(canonical_chromosomes))
            .with_columns(pl.col("chr_name").cast(pl.Enum(canonical_chromosomes)))
            .select(["chr_name", "chr_position", "ref_allele", "alt_allele", "variant_id", "variant_name"])
        )

        return cls(gen_file, obs, variants, genome_build)
