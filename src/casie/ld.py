from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from scipy.sparse import csr_array

from .build import GenomeBuild


@dataclass(repr=False, frozen=True, slots=True)
class LDMatrix:
    ld: csr_array
    ld_type: Literal["covariance", "correlation"]
    _variants: pl.DataFrame
    genome_build: GenomeBuild

    @classmethod
    def from_predictdb(
        cls, 
        cov: str | Path, 
        *,
        genome_build: str | GenomeBuild
    ) -> LDMatrix:
        ...