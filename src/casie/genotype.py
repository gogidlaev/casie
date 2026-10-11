import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import polars as pl
from pgenlib import PgenReader, PvarReader

from .build import GenomeBuild


class GenotypeFile(ABC):
    @abstractmethod
    def read_obs(self) -> pl.DataFrame:
        """
        Read the sample metadata from the genotype file.
        """
        raise NotImplementedError

    @abstractmethod
    def scan_variants(self) -> pl.LazyFrame:
        """
        Scan the variant metadata from the genotype file.
        """
        raise NotImplementedError

    @abstractmethod
    def read_dosages(self, variant_idx: np.ndarray) -> np.ndarray:
        """
        Read the dosages for the specified variants.
        """
        raise NotImplementedError


@dataclass
class PgenFile(GenotypeFile):
    pgen: Path
    psam: Path
    pvar: Path

    _pgen_reader: PgenReader | None = field(default=None, init=False, repr=False)
    _pvar_reader: PvarReader | None = field(default=None, init=False, repr=False)

    _sample_subset: np.ndarray | None = field(default=None, init=False, repr=False)
    _sample_ct: int | None = field(default=None, init=False, repr=False)

    def read_obs(self) -> pl.DataFrame:
        """
        Read and validate the sample metadata from the .psam file.
        """
        field_separator = re.compile(r"[ \t]+")
        fam_columns = ("FID", "IID", "PAT", "MAT", "SEX")
        spool_max_size = 8 * 1024 * 1024

        source = self.psam
        header: list[str] | None = None
        first_blank_line: int | None = None
        sample_rows = 0
        first_identity_lines: dict[tuple[str, str, str], int] = {}
        iid_index: int | None = None
        fid_index: int | None = None
        sid_index: int | None = None

        def parse_error(line_number: int | None, reason: str) -> ValueError:
            location = f"{source}:"
            if line_number is not None:
                location += f" line {line_number}:"
            return ValueError(f"{location} {reason}")

        with source.open("rb") as handle, tempfile.SpooledTemporaryFile(
            mode="w+b",
            max_size=spool_max_size,
        ) as normalized:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise parse_error(line_number, "invalid UTF-8") from error

                content = line.strip(" \t\r\n")
                fields = field_separator.split(content) if content else []

                if not fields:
                    first_blank_line = first_blank_line or line_number
                    continue
                if first_blank_line is not None:
                    raise parse_error(
                        first_blank_line,
                        "blank lines are only permitted at end of file",
                    )

                if header is None:
                    if fields[0].startswith("#") and fields[0] not in {
                        "#FID",
                        "#IID",
                    }:
                        continue
                    if fields[0] in {"#FID", "#IID"}:
                        header = [fields[0][1:], *fields[1:]]

                        seen_names: set[str] = set()
                        for name in header:
                            if name in seen_names:
                                raise parse_error(
                                    line_number,
                                    f"duplicate column name {name!r}",
                                )
                            seen_names.add(name)

                        if "FID" in seen_names:
                            if header[0] != "FID":
                                raise parse_error(
                                    line_number,
                                    "FID must be the first column",
                                )
                            if len(header) < 2 or header[1] != "IID":
                                raise parse_error(
                                    line_number,
                                    "IID must be column 2 when FID is present",
                                )
                        elif header[0] != "IID":
                            raise parse_error(
                                line_number,
                                "IID must be the first column",
                            )

                        if (
                            "SID" in seen_names
                            and header.index("SID") != header.index("IID") + 1
                        ):
                            raise parse_error(
                                line_number,
                                "SID must immediately follow IID",
                            )

                        iid_index = header.index("IID")
                        fid_index = header.index("FID") if "FID" in header else None
                        sid_index = header.index("SID") if "SID" in header else None
                        continue

                    if len(fields) < len(fam_columns):
                        raise parse_error(
                            line_number,
                            "headerless PSAM requires at least 5 fields",
                        )
                    header = [
                        *fam_columns,
                        *(
                            f"PHENO{index}"
                            for index in range(
                                1,
                                len(fields) - len(fam_columns) + 1,
                            )
                        ),
                    ]
                    iid_index = header.index("IID")
                    fid_index = header.index("FID")

                if fields[0].startswith("#"):
                    raise parse_error(
                        line_number,
                        "comments are only permitted before the primary header",
                    )
                if len(fields) != len(header):
                    raise parse_error(
                        line_number,
                        f"expected {len(header)} fields, found {len(fields)}",
                    )

                assert iid_index is not None
                iid = fields[iid_index]
                if iid == "0":
                    raise parse_error(line_number, "IID cannot be '0'")
                fid = fields[fid_index] if fid_index is not None else "0"
                sid = fields[sid_index] if sid_index is not None else "0"
                identity = (fid, iid, sid)
                if identity in first_identity_lines:
                    raise parse_error(
                        line_number,
                        "duplicate sample ID "
                        f"(first occurrence on line "
                        f"{first_identity_lines[identity]})",
                    )
                first_identity_lines[identity] = line_number
                normalized.write("\t".join(fields).encode("utf-8") + b"\n")
                sample_rows += 1

            if header is None or sample_rows == 0:
                raise parse_error(None, "contains no sample rows")

            normalized.seek(0)
            return pl.read_csv(
                normalized,
                has_header=False,
                new_columns=header,
                schema={name: pl.String for name in header},
                separator="\t",
                quote_char=None,
                null_values=None,
                empty_string_is_null=False,
                infer_schema=False,
                ignore_errors=False,
                try_parse_dates=False,
                truncate_ragged_lines=False,
                raise_if_empty=True,
            ).with_row_index("_obs_idx")

    def scan_variants(self):
        return (
            pl.scan_csv(
                self.pvar,
                separator="\t",
                comment_prefix="##",
                schema_overrides={
                    "#CHROM": pl.String,
                    "POS": pl.UInt64,
                    "ID": pl.String,
                    "REF": pl.String,
                    "ALT": pl.String
                },
            )
            .with_row_index("_variant_idx")
            .select(
                "_variant_idx",
                pl.col("#CHROM").alias("chr_name"),
                pl.col("POS").alias("chr_position"),
                pl.col("REF").alias("ref_allele"),
                pl.col("ALT").alias("alt_allele"),
                pl.col("ID").alias("variant_name")
            )
        )

    def open(self, obs_idx: np.ndarray | None = None) -> Self:
        # Sample metadata can be filtered before opening the .pgen, so we must take care to pass the correct subset. It also should help with memory management.
        if self._pgen_reader is not None or self._pvar_reader is not None:
            raise RuntimeError("PgenFile already open")

        self._sample_subset = obs_idx
        return self

    def __enter__(self) -> Self:
        if self._pgen_reader is not None or self._pvar_reader is not None:
            raise RuntimeError("PgenFile already open")

        sample_subset = self._sample_subset

        try:
            # Passing opened .pvar is required for opening .pgen if the data contains multiallelic variants.
            pvar_reader = PvarReader(
                os.fsencode(self.pvar),
                omit_chrom=True,
                omit_pos=True
            )
            self._pvar_reader = pvar_reader

            pgen_reader = PgenReader(
                os.fsencode(self.pgen),
                sample_subset=self._sample_subset,
                pvar=self._pvar_reader
            )
            self._pgen_reader = pgen_reader

            self._sample_ct = (
                pgen_reader.get_raw_sample_ct()
                if sample_subset is None
                else len(sample_subset)
            )

        except Exception:
            self.close()
            raise

        return self

    def __exit__(
        self, 
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> bool:
        self.close()
        return False

    def close(self) -> None:
        pgen_reader = self._pgen_reader
        pvar_reader = self._pvar_reader

        self._pgen_reader = None
        self._pvar_reader = None
        self._sample_subset = None
        self._sample_ct = None

        try:
            if pgen_reader is not None:
                pgen_reader.close()
        finally:
            if pvar_reader is not None:
                pvar_reader.close()

    def read_dosages(self, variant_idx: np.ndarray) -> np.ndarray:
        if (
            self._pgen_reader is None
            or self._pvar_reader is None
            or self._sample_ct is None
        ):
            raise RuntimeError("PgenFile is not open")

        dosages = np.empty((variant_idx.shape[0], self._sample_ct), dtype=np.float64)

        self._pgen_reader.read_dosages_list(variant_idx, dosages)

        return dosages


@dataclass(frozen=True, slots=True, repr=False)
class GenotypeDataset:
    gen_files: list[PgenFile]
    gen_file_map: pl.DataFrame
    _obs: pl.DataFrame
    _variants: pl.LazyFrame
    genome_build: GenomeBuild

    @classmethod
    def from_pgen(
        cls, 
        pgen: str,
        *,
        genome_build: str | GenomeBuild
    ):
        genome_build = GenomeBuild.parse_from(genome_build)

        gen_files = []
        pgen_files = sorted(Path(pgen_file) for pgen_file in glob(pgen))

        # Handle the case where no .pgen files are found
        if not pgen_files:
            raise FileNotFoundError("No .pgen files matched")
        # Handle the case where the pattern matches files that are not .pgen or are not files
        if invalid := [path for path in pgen_files if path.suffix != ".pgen" or not path.is_file()]:
            raise ValueError(f"Invalid .pgen matches: {invalid}")

        # Find the corresponding .psam and .pvar files for each .pgen file
        # Construct the PgenFile objects
        for pgen_file in pgen_files:
            psam_file = pgen_file.with_suffix(".psam")

            pvar_files = [pgen_file.with_suffix(".pvar"), pgen_file.with_suffix(".pvar.zst")]
            pvar_files = [pvar_file for pvar_file in pvar_files if pvar_file.exists()]
            if not pvar_files:
                raise FileNotFoundError(f"No accompanying .pvar file found for {pgen_file}")
            if len(pvar_files) > 1:
                raise ValueError(f"Multiple accompanying .pvar files found for {pgen_file}: {pvar_files}")
            pvar_file = pvar_files[0]

            gen_files.append(PgenFile(pgen_file, psam_file, pvar_file))

        raw_obs = [gen_file.read_obs() for gen_file in gen_files]

        obs_ids = []
        obs_groups = []
        obs = []

        for raw in raw_obs:
            id_cols = [col for col in raw.columns if col in ("FID", "IID", "SID")]

            ids = raw.select(id_cols)
            match = next(
                (idx for idx, other in enumerate(obs_ids) if other.equals(ids)),
                None,
            )
            if match is None:
                match = len(obs_ids)
                obs_ids.append(ids)
                obs.append(raw.with_columns(_obs_group_idx=pl.lit(match)))
            obs_groups.append(match)

        variants = []

        if len(obs) == 1:
            # This dataset is split by variants, so we simply read the variants from all gen_files and assign unique group indices
            variants_groups = range(len(gen_files))
            for i, gen_file in enumerate(gen_files):
                variants.append(gen_file.scan_variants().with_columns(_variant_group_idx=pl.lit(i)))
        elif len(obs) == len(gen_files):
            variants_groups = [0] * len(gen_files)
            variants.append(gen_files[0].scan_variants().with_columns(_variant_group_idx=pl.lit(0)))
        else:
            variants_fingerprints = []
            variants_groups = []

            for i, gen_file in enumerate(gen_files):
                pvar_stat = gen_file.pvar.stat()
                pvar_fingerprint = (pvar_stat.st_size, pvar_stat.st_mtime_ns)

                match = next(
                    (idx for idx, other in enumerate(variants_fingerprints) if other == pvar_fingerprint),
                    None,
                )
                if match is None:
                    match = len(variants_fingerprints)
                    variants_fingerprints.append(pvar_fingerprint)

                    variants_ = gen_file.scan_variants().with_columns(_variant_group_idx=pl.lit(match))
                    variants.append(variants_)
                variants_groups.append(match)

        gen_file_map = pl.DataFrame({
            "file_idx": pl.arange(0, len(gen_files), dtype=pl.UInt32, eager=True),
            "obs_group_idx": pl.Series(obs_groups, dtype=pl.UInt32),
            "variant_group_idx": pl.Series(variants_groups, dtype=pl.UInt32)
        })

        return cls(gen_files, gen_file_map, pl.concat(obs), pl.concat(variants), genome_build)
