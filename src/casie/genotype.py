from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from pgenlib import PgenReader, PvarReader


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
                dosages = np.ndarray(
                    shape=(variant_idx.len(), obs_idx.len()), 
                    dtype=np.float32
                )
                pgen_reader.read_dosages_list(variant_idx.to_numpy(), dosages)
                
                return dosages
