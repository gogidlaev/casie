from __future__ import annotations

from dataclasses import dataclass, replace

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