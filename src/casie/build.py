from enum import StrEnum


class GenomeBuild(StrEnum):
    GRCH38 = "GRCh38"
    GRCH37 = "GRCh37"

    @classmethod
    def parse_from(cls, genome_build: str):
        if isinstance(genome_build, str):
            match genome_build.lower():
                case "grch38" | "hg38" | "b38":
                    return cls("GRCh38")
                case "grch37" | "hg19" | "b37":
                    return cls("GRCh37")
                case _:
                    raise ValueError(f"Unknown genome build: {genome_build}")
        else:
            raise TypeError(
                "GenomeBuild can only be parsed from string, "
                f"not {type(genome_build)}"
            )
            