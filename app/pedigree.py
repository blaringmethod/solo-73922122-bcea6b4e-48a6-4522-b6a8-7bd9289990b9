"""Pedigree (trio) Mendelian incompatibility review on a cohort VCF.

The reviewer picks three *distinct* cohort samples as father, mother and
child and scans every site of the already-generated cohort VCF.

Per site, for each person the Phred-scaled genotype likelihoods are
converted to linear likelihoods ``P = 10 ** (-PL / 10)`` and normalized over
the diploid genotype space, following the VCF triangular PL order for
multi-ALT sites::

    (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3), ...

The joint probability of every father/mother/child genotype combination is
the product of the three marginals. A combination is Mendelian-compatible
when the child can inherit one allele from each parent; the incompatibility
probability is the summed probability of all incompatible combinations.
Sites reaching ``config.PEDIGREE_INCOMPATIBILITY_THRESHOLD`` are returned
for manual review, with the coordinate, the three GT strings and the
probability.

A site is **undecidable** -- never silently counted as consistent -- when
any of the three people has a missing GT or an unusable PL field (PL
missing, no numeric value, or a length that is not a diploid triangular
number). Undecidable sites are reported separately and excluded from both
the incompatible and consistent counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pysam


class PedigreeError(ValueError):
    """Raised when a trio review cannot be started (bad input/state)."""


@dataclass
class SiteReview:
    chrom: str
    pos: int
    ref: str
    alts: tuple[str, ...]
    father_gt: str
    mother_gt: str
    child_gt: str
    incompatibility_probability: float


@dataclass
class UndecidableSite:
    chrom: str
    pos: int
    reason: str
    father_gt: str
    mother_gt: str
    child_gt: str


@dataclass
class TrioReport:
    father: str
    mother: str
    child: str
    selection_version: int
    threshold: float
    total_sites: int = 0
    decidable_sites: int = 0
    incompatible_sites: list[SiteReview] = field(default_factory=list)
    undecidable_sites: list[UndecidableSite] = field(default_factory=list)

    @property
    def consistent_sites(self) -> int:
        return self.decidable_sites - len(self.incompatible_sites)

    def to_dict(self) -> dict:
        return {
            "father": self.father,
            "mother": self.mother,
            "child": self.child,
            "selection_version": self.selection_version,
            "threshold": self.threshold,
            "total_sites": self.total_sites,
            "decidable_sites": self.decidable_sites,
            "consistent_sites": self.consistent_sites,
            "incompatible_count": len(self.incompatible_sites),
            "undecidable_count": len(self.undecidable_sites),
            "incompatible_sites": [
                {
                    "chrom": s.chrom,
                    "pos": s.pos,
                    "ref": s.ref,
                    "alts": list(s.alts),
                    "father_gt": s.father_gt,
                    "mother_gt": s.mother_gt,
                    "child_gt": s.child_gt,
                    "incompatibility_probability": s.incompatibility_probability,
                }
                for s in self.incompatible_sites
            ],
            "undecidable_sites": [
                {
                    "chrom": s.chrom,
                    "pos": s.pos,
                    "reason": s.reason,
                    "father_gt": s.father_gt,
                    "mother_gt": s.mother_gt,
                    "child_gt": s.child_gt,
                }
                for s in self.undecidable_sites
            ],
        }


# --------------------------------------------------------------------------- #
# PL / genotype helpers
# --------------------------------------------------------------------------- #
def triangular_allele_count(pl_length: int) -> int | None:
    """Invert ``n*(n+1)/2 = pl_length``; return None when not triangular.

    A diploid PL vector over ``n`` alleles always has the triangular count
    of unordered genotypes, so its length identifies the genotype space even
    at no-ALT sites where pysam reports only one allele.
    """
    if pl_length <= 0:
        return None
    discriminant = 1 + 8 * pl_length
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        return None
    n = (root - 1) // 2
    return n if n >= 1 else None


def genotype_posteriors(pl_values: tuple) -> list[float] | None:
    """Convert PL to normalized posteriors ``10**(-PL/10) / sum``.

    Returns None when the vector has no numeric entry or its length is not a
    diploid triangular number. Missing entries (``.``) are treated as zero
    likelihood, which is the VCF convention for an unsupported genotype.
    """
    n_alleles = triangular_allele_count(len(pl_values))
    if n_alleles is None:
        return None
    weights: list[float] = []
    has_value = False
    for value in pl_values:
        if value is None:
            weights.append(0.0)
            continue
        has_value = True
        weights.append(10.0 ** (-float(value) / 10.0))
    if not has_value:
        return None
    total = math.fsum(weights)
    if total <= 0.0:
        return None
    return [w / total for w in weights]


def genotype_alleles(genotype_index: int, n_alleles: int) -> tuple[int, int]:
    """Decode a VCF triangular PL index into its unordered allele pair."""
    # Largest j with j*(j+1)/2 <= index; i = index - j*(j+1)/2.
    j = int((math.isqrt(8 * genotype_index + 1) - 1) // 2)
    i = genotype_index - j * (j + 1) // 2
    return i, j


def is_mendelian_compatible(
    father_pair: tuple[int, int],
    mother_pair: tuple[int, int],
    child_pair: tuple[int, int],
) -> bool:
    """True when the child can inherit one allele from each parent.

    Every assignment of the two child alleles to a paternal/maternal origin
    is tried, including equal-allele (homozygous) cases.
    """
    child_a, child_b = child_pair
    father_set = set(father_pair)
    mother_set = set(mother_pair)
    return (
        child_a in father_set and child_b in mother_set
    ) or (
        child_b in father_set and child_a in mother_set
    )


def incompatibility_probability(
    father_post: list[float],
    mother_post: list[float],
    child_post: list[float],
    n_alleles: int,
) -> float:
    """Sum the joint probability of every incompatible trio combination."""
    n_genotypes = n_alleles * (n_alleles + 1) // 2
    pairs = [genotype_alleles(g, n_alleles) for g in range(n_genotypes)]
    incompatible_mass = 0.0
    for gf, pf in enumerate(father_post):
        if pf == 0.0:
            continue
        father_pair = pairs[gf]
        for gm, pm in enumerate(mother_post):
            if pm == 0.0:
                continue
            mother_pair = pairs[gm]
            parent_weight = pf * pm
            for gc, pc in enumerate(child_post):
                if pc == 0.0:
                    continue
                if not is_mendelian_compatible(
                    father_pair, mother_pair, pairs[gc]
                ):
                    incompatible_mass += parent_weight * pc
    # Guard against tiny floating-point drift outside [0, 1].
    return min(1.0, max(0.0, incompatible_mass))


def format_gt(sample_data) -> str:
    """Render a sample GT as VCF text (``./.`` for missing), preserving phase."""
    gt = sample_data.get("GT")
    if gt is None or None in gt:
        return "./."
    separator = "|" if sample_data.phased else "/"
    return separator.join(str(allele) for allele in gt)


# --------------------------------------------------------------------------- #
# Trio review
# --------------------------------------------------------------------------- #
def _site_undecidable(
    record: pysam.VariantRecord,
    people: tuple[str, str, str],
    reason: str,
) -> UndecidableSite:
    father, mother, child = people
    return UndecidableSite(
        chrom=record.chrom,
        pos=record.pos,
        reason=reason,
        father_gt=format_gt(record.samples[father]),
        mother_gt=format_gt(record.samples[mother]),
        child_gt=format_gt(record.samples[child]),
    )


def review_trio(
    vcf_path: str,
    father: str,
    mother: str,
    child: str,
    selection_version: int,
    threshold: float,
) -> TrioReport:
    """Scan a cohort VCF and collect Mendelian-incompatible review sites."""
    people = (father, mother, child)
    report = TrioReport(
        father=father,
        mother=mother,
        child=child,
        selection_version=selection_version,
        threshold=threshold,
    )

    vcf = pysam.VariantFile(vcf_path)
    try:
        header_samples = set(vcf.header.samples)
        missing = [name for name in people if name not in header_samples]
        if missing:
            raise PedigreeError(
                "队列 VCF 中不存在样本：" + "、".join(missing)
            )

        for record in vcf:
            report.total_sites += 1

            # Rule: any missing GT makes the site undecidable, never
            # silently consistent.
            missing_gt = [
                name
                for name in people
                if (gt := record.samples[name].get("GT")) is None
                or None in gt
            ]
            if missing_gt:
                report.undecidable_sites.append(
                    _site_undecidable(
                        record,
                        people,
                        "GT 缺失：" + "、".join(missing_gt),
                    )
                )
                continue

            posteriors: list[list[float]] = []
            pl_problem: str | None = None
            for name in people:
                pl_values = record.samples[name].get("PL")
                if pl_values is None:
                    pl_problem = f"PL 缺项：{name}"
                    break
                post = genotype_posteriors(tuple(pl_values))
                if post is None:
                    pl_problem = f"PL 缺项或长度不符合二倍体三角布局：{name}"
                    break
                posteriors.append(post)
            if pl_problem is not None:
                report.undecidable_sites.append(
                    _site_undecidable(record, people, pl_problem)
                )
                continue

            n_father = triangular_allele_count(
                len(record.samples[father].get("PL"))
            )
            n_mother = triangular_allele_count(
                len(record.samples[mother].get("PL"))
            )
            n_child = triangular_allele_count(
                len(record.samples[child].get("PL"))
            )
            if not (n_father == n_mother == n_child):
                report.undecidable_sites.append(
                    _site_undecidable(
                        record,
                        people,
                        "三人 PL 基因型空间（等位基因数）不一致，无法联合计算",
                    )
                )
                continue

            n_alleles = n_father
            probability = incompatibility_probability(
                posteriors[0], posteriors[1], posteriors[2], n_alleles
            )
            report.decidable_sites += 1

            if probability >= threshold:
                report.incompatible_sites.append(
                    SiteReview(
                        chrom=record.chrom,
                        pos=record.pos,
                        ref=record.ref,
                        alts=tuple(record.alts or ()),
                        father_gt=format_gt(record.samples[father]),
                        mother_gt=format_gt(record.samples[mother]),
                        child_gt=format_gt(record.samples[child]),
                        incompatibility_probability=probability,
                    )
                )
    finally:
        vcf.close()

    return report
