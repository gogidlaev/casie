from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from scipy.sparse import coo_array, csc_array

from .build import GenomeBuild
from .variants import VariantFrame


@dataclass(frozen=True, slots=True, repr=False)
class PGSCollection:
    """A class for representing a collection of sparse polygenic scores.
    
    Use `from_predictdb` to load a PredictDB database. To construct a collection from custom weights, use `from_long_weights`.
    Attributes
    ----------
    weights : scipy.sparse.csc_array
        PGS weights in sparse CSC format with shape ``(n_variants, n_features)``.
    features : polars.DataFrame
        Score metadata, with rows corresponding to weight matrix columns.
        Each score is identified by (collection_id, feature_id).
    variants : polars.DataFrame
        Variant metadata, with rows corresponding to weight matrix rows.
    collections : polars.DataFrame
        Collection metadata
    standardized : bool
        Whether wights apply to standardized genotypes (True) or raw dosages (False). 
        This does not modify the weights but determines their use for downstream methods
        such as summary-level TWAS.
    genome_build : casie.GenomeBuild
        Reference genome assembly for the variant coordinates.
    """
    weights: csc_array
    _features: pl.DataFrame
    _variants: VariantFrame
    _collections: pl.DataFrame
    standardized: bool

    @classmethod
    def from_long_weights(
        cls,
        weights: pl.DataFrame | pl.LazyFrame,
        *,
        standardized: bool,
        genome_build: str | GenomeBuild,
        collection_id: str,
        feature_id: str,
        effect_weight: str,
        chr_name: str = "chr_name",
        chr_position: str = "chr_position",
        other_allele: str = "other_allele",
        effect_allele: str = "effect_allele",
        variant_name: str | None = None,
        features: pl.DataFrame | None = None,
        dataset: str | None = None,
        context: str | None = None,
        modality: str | None = None,
        method: str | None = None
    ) -> PGSCollection:
        """Construct PGSCollection from a long-form dataframe of weights.

        Parameters
        ----------
        weights : polars.DataFrame | polars.LazyFrame
            Long-form dataframe with weights, variants and feature IDs
        standardized : bool
            Whether the weights in the collection are with respect to raw (False)
            or standardized (True) genotypes
        genome_build : str | casie.GenomeBuild
            Reference genome assembly for the variant coordinates. If a string, must be 
            parseable by `GenomeBuild.parse_from` method.
        collection_id : str
            Identifier of the collection. Together with feature_id, identifies
            each score. Distinct collections must use distinct identifiers.
        feature_id : str
            Name of the column that contains feature IDs. If 'features' are provided, both
            dataframes must use the same column name to identify features. If it's not "feature_id", the
            specified column will only be copied to "feature_id" column 
        effect_weight : str
            Name of the column that contains the weight of the effect allele
        chr_name : str, default "chr_name"
            Name of the column that contains chromosome names
        chr_position : str, default "chr_position"
            Name of the column that contains chromosomal positions
        other_allele : str, default "other_allele"
            Name of the column that contains non-effect allele. This is commonly (but now always)
            the reference allele
        effect_allele : str, default "effect_allele"
            Name of the column that contains the effect allele, i.e. the allele which counts for the PGS calculation.
            This is commonly (but not always) the alternative allele
        variant_name : str, optional
            Name of the column that contains variant names (such as rsID)
        features : pl.DataFrame, optional
            Optional dataframe with feature metadata
        dataset : str, optional
            Dataset name such as GTEx or CMC
        context : str, optional
            Model context such as specific tissue or cell type
        modality : str, optional
            Type of the measured phenotype such as expression or splicing
        method : str, optional
            Method used to calculate PGS such as enet or mashr. Because it's a collection-level value, FUSION format is supported provisionally for now

        Notes
        -----
        Each call represents one dataset, context, modality, and method. These arguments are optional but recommended.
        If columns with these names are in the supplied feature metadata, each must contain exactly
        one distinct value and no nulls, unless selected as the feature ID column.
        Omitted arguments are inferred from those columns; supplied arguments must
        agree with them. Resolved values populate collection and feature metadata.
        Score identity is determined by (collection_id, feature_id). The weights are 
        filtered for non-zero effect weights and zero-predictor features with a warning; thus,
        assembled metadata might have slightly fewer features than the original source.
        """
        if genome_build is None:
            raise ValueError("genome_build is required")
        genome_build = GenomeBuild.parse_from(genome_build)

        if collection_id is None:
            raise ValueError("collection_id is required")

        collection = {"collection_id": collection_id}
        collection_fields = {
            "dataset": dataset,
            "context": context,
            "modality": modality,
            "method": method,
        }
        for key, value in collection_fields.items():
            if features is not None and key != feature_id and key in features.columns:
                metadata_column = features.get_column(key)
                if metadata_column.null_count():
                    raise ValueError(
                        f"Feature metadata column {key!r} contains null values"
                    )
                metadata_values = metadata_column.unique()
                if len(metadata_values) != 1:
                    raise ValueError(
                        f"Feature metadata column {key!r} must contain exactly one "
                        "unique non-null value per collection"
                    )
                metadata_value = metadata_values.item()
                if value is not None and value != metadata_value:
                    raise ValueError(
                        f"{key}={value!r} conflicts with feature metadata "
                        f"value {metadata_value!r}"
                    )
                value = metadata_value
            if value is not None:
                collection[key] = value

        collection_keys = [key for key in collection_fields if key in collection]

        required_cols = [feature_id, chr_name, chr_position, other_allele, effect_allele, effect_weight]
        if variant_name is not None:
            required_cols.append(variant_name)

        weight_columns = (
            weights.collect_schema().names()
            if isinstance(weights, pl.LazyFrame)
            else weights.columns
        )
        missing_cols = set(required_cols).difference(weight_columns)
        
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not in weights")

        weight_cols = {
            "feature_id": pl.col(feature_id).cast(pl.String),
            "effect_weight": pl.col(effect_weight),
            "chr_name": pl.col(chr_name).cast(pl.String),
            "chr_position": pl.col(chr_position),
            "other_allele": pl.col(other_allele).cast(pl.String),
            "effect_allele": pl.col(effect_allele).cast(pl.String),
        }
        if variant_name is not None:
            weight_cols["variant_name"] = pl.col(variant_name)

        # Capture source fields before generating columns that may share their names.
        weights = weights.select(
            expr.alias(name) for name, expr in weight_cols.items()
        )
        weights = VariantFrame.normalize_chr(weights)
        weights = VariantFrame.normalize_position(weights)
        weights = VariantFrame.add_variant_id(weights)
        if isinstance(weights, pl.LazyFrame):
            weights = weights.collect()

        variant_cols = [
            "chr_name", "chr_position", "other_allele", "effect_allele",
            "variant_id",
        ]
        if variant_name is not None:
            variant_cols.append("variant_name")

        variants = weights.select(variant_cols).unique("variant_id")
        variants = VariantFrame.sort_variants(variants).with_row_index("_variant_idx")
        variants = VariantFrame(
            variants.collect() if isinstance(variants, pl.LazyFrame) else variants,
            a1="other_allele",
            a2="effect_allele",
            genome_build=genome_build,
            hide_cols=["_variant_idx"],
        )

        if weights.select(["feature_id", "variant_id"]).is_duplicated().any():
            raise ValueError("Duplicate feature x variant combinations in weights")

        effect_weights = weights.get_column("effect_weight")

        if effect_weights.null_count():
            raise ValueError(f"{effect_weight} column contains null weights")
        if not effect_weights.is_finite().all():
            raise ValueError(f"{effect_weight} column contains NaN or infinite weights")

        # Supplied feature metadata still uses the caller's source column name.
        weight_feature_ids = weights.select(
            pl.col("feature_id").alias(feature_id)
        ).unique(maintain_order=True)

        if weight_feature_ids.get_column(feature_id).null_count():
            raise ValueError("Missing feature ids in weights are not allowed")

        if features is None:
            features_frame = weight_feature_ids
        else:
            if feature_id not in features.columns:
                raise ValueError(f"Column {feature_id} must be present in features")
            features = features.with_columns(pl.col(feature_id).cast(pl.String))
            metadata_ids = features.get_column(feature_id)
            
            if metadata_ids.null_count():
                raise ValueError("Missing feature ids in features are not allowed")
            if metadata_ids.is_duplicated().any():
                raise ValueError("Duplicate feature ids in features are not allowed")

            missing_features = weight_feature_ids.join(
                features.select(feature_id),
                on=feature_id,
                how="anti",
            )
            if not missing_features.is_empty():
                raise ValueError(f"{missing_features.height} features in weights are missing from features metadata")

            features_frame = features

        n_weights_before = weights.height
        weights = weights.filter(pl.col("effect_weight") != 0)
        n_zero_weights = n_weights_before - weights.height
        
        if n_zero_weights:
            warnings.warn(
                f"Collection {collection_id!r}: removed "
                f"{n_zero_weights} zero-weight rows.",
                UserWarning,
                stacklevel=2,
            )

        active_feature_ids = weights.select(
            pl.col("feature_id").alias(feature_id)
        ).unique()

        n_features_before = features_frame.height
        features_frame = features_frame.join(
            active_feature_ids,
            on=feature_id,
            how="semi",
            maintain_order="left",
        )
        n_empty_features = n_features_before - features_frame.height

        if n_empty_features:
            warnings.warn(
                f"Collection {collection_id!r}: removed "
                f"{n_empty_features} features with no nonzero weights.",
                UserWarning,
                stacklevel=2,
            )

        n_variants = weights.group_by("feature_id").len("n_variants")

        features_frame = features_frame.with_row_index("_pgs_idx")
        feature_index = features_frame.select(
            pl.col(feature_id).alias("feature_id"), "_pgs_idx"
        )

        features_frame = (
            features_frame
            .with_columns(pl.col(feature_id).alias("feature_id"))
            .drop("n_variants", strict=False)
            .join(n_variants, on="feature_id", how="inner", maintain_order="left")
        )

        n_variant_cols = ["n_variants"]
        if "n_variants_reported" in features_frame.columns:
            n_variant_cols = ["n_variants_reported"] + n_variant_cols
        first_cols = [
            "_collection_idx", "_pgs_idx", "collection_id", "feature_id",
            *collection_keys, *n_variant_cols
        ]

        features_frame = features_frame.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("_collection_idx"),
            pl.lit(collection_id).alias("collection_id"),
            *[pl.lit(collection[key]).alias(key) for key in collection_keys],
        ).select(*first_cols, pl.exclude(first_cols))

        weights = (
            weights
            .select(["variant_id", "feature_id", "effect_weight"])
            .join(variants.frame.select(["_variant_idx", "variant_id"]), on="variant_id")
            .join(feature_index, on="feature_id", validate="m:1")
            .select(["_variant_idx", "_pgs_idx", "effect_weight"])
        )
        weights_csc = coo_array(
            (
                weights["effect_weight"].to_numpy(),
                (
                    weights["_variant_idx"].to_numpy(),
                    weights["_pgs_idx"].to_numpy(),
                ),
            ),
            shape=(variants.frame.height, features_frame.height),
        ).tocsc()

        collections_frame = pl.DataFrame({
            **collection,
            "n_features": features_frame.height,
        }).with_row_index("_collection_idx")
        
        return cls(
            weights=weights_csc, 
            _features=features_frame, 
            _variants=variants, 
            _collections=collections_frame, 
            standardized=standardized
        )

    @classmethod
    def from_predictdb(
        cls,
        db: str | Path,
        genome_build: str | GenomeBuild | None = None,
        collection_id: str | None = None,
        dataset: str | None = None,
        context: str | None = None,
        modality: str | None = None,
        method: str | None = None
    ) -> PGSCollection:
        """
        Load a PredictDB database as a PGSCollection

        Parameters
        ----------
        db : str | Path
            Path to the PredictDB database
        genome_build : str | casie.GenomeBuild
            Reference genome assembly for the variant coordinates. If a string, must be 
            parseable by `GenomeBuild.parse_from` method.
        collection_id : str, optional
            Identifier of the collection. Together with feature_id, identifies
            each score. If not provided, the name of the database if used as the collection ID.
            Distinct collections must use distinct identifiers.
        dataset : str, optional
            Dataset name such as GTEx or CMC
        context : str, optional
            Model context such as specific tissue or cell type
        modality : str, optional
            Type of the measured phenotype such as expression or splicing
        method : str, optional
            Method used to calculate PGS such as enet or mashr
        """
        from contextlib import closing

        import sqlite3

        db_path = Path(db)
        if not db_path.exists():
            raise FileNotFoundError(f"PredictDB database not found: {db_path}")

        if collection_id is None:
            collection_id = db_path.stem

        with closing(sqlite3.connect(db_path)) as conn:
            extra = pl.read_database("SELECT * FROM extra", conn)
            weights = pl.read_database("SELECT * FROM weights", conn)
        
        extra = extra.rename({
            "gene": "gene_id",
            "genename": "gene_name",
            "pred.perf.R2": "pred_r2",
            "pred.perf.pval": "pred_pval",
            "n.snps.in.model": "n_variants_reported"
        })
        required_cols = ["gene_id", "gene_name", "pred_r2", "pred_pval", "n_variants_reported"]
        extra = extra.select(pl.col(required_cols), pl.exclude(required_cols))

        weights = (
            weights
            .with_columns(
                pl.col("gene").alias("gene_id"),
                pl.col("varID").str.split("_").list.to_struct(
                    fields=["chr_name", "chr_position", "varid_other_allele", "varid_effect_allele", "genome_build"]
                ).struct.unnest(),
                pl.col("ref_allele").alias("other_allele"),
                pl.col("eff_allele").alias("effect_allele")
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

        return cls.from_long_weights(
            weights,
            standardized=False,
            genome_build=genome_build,
            feature_id="gene_id",
            effect_weight="weight",
            variant_name="rsid",
            features=extra,
            collection_id=collection_id,
            dataset=dataset,
            context=context,
            modality=modality,
            method=method
        )

    @property
    def features(self):
        return self._features.drop(["_collection_idx", "_pgs_idx"])

    @property
    def variants(self):
        if self._variants.hide_cols:
            return self._variants.frame.drop(self._variants.hide_cols)
        else:
            return self._variants.frame

    @property
    def genome_build(self):
        return self._variants.genome_build
    
    @property
    def collections(self):
        return self._collections.drop(["_collection_idx"])
