"""Tests for pedigree (trio) Mendelian-incompatibility review.

Covers the PL math in isolation and the full HTTP workflow, including the
integrity rule that a review must never run against a cohort result made
stale by a later source-selection change.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.pedigree import (
    PedigreeError,
    genotype_alleles,
    genotype_posteriors,
    incompatibility_probability,
    is_mendelian_compatible,
    review_trio,
    triangular_allele_count,
)

HEADER = """##fileformat=VCFv4.2
##reference=GRCh38.demo.1.0
##contig=<ID=chr1,length=100000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}
"""


def write_vcf(path: Path, samples: list[str], rows: list[str]) -> Path:
    path.write_text(
        HEADER.format(samples="\t".join(samples)) + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Pure math helpers
# --------------------------------------------------------------------------- #
def test_triangular_allele_count():
    assert triangular_allele_count(1) == 1
    assert triangular_allele_count(3) == 2
    assert triangular_allele_count(6) == 3
    assert triangular_allele_count(10) == 4
    assert triangular_allele_count(2) is None
    assert triangular_allele_count(5) is None
    assert triangular_allele_count(0) is None


def test_genotype_posteriors_normalization():
    # Biallelic PL 0,30,250 -> weights 1, 0.001, 1e-25.
    post = genotype_posteriors((0, 30, 250))
    assert post is not None
    assert post[0] == pytest.approx(1 / 1.001)
    assert post[1] == pytest.approx(0.001 / 1.001)
    assert sum(post) == pytest.approx(1.0)


def test_genotype_posteriors_rejects_bad_vectors():
    assert genotype_posteriors(()) is None
    assert genotype_posteriors((None, None)) is None
    # Length 2 is not a diploid triangular number.
    assert genotype_posteriors((0, 10)) is None


def test_genotype_alleles_follows_vcf_layout():
    # (00),(01),(11),(02),(12),(22)
    assert genotype_alleles(0, 3) == (0, 0)
    assert genotype_alleles(1, 3) == (0, 1)
    assert genotype_alleles(2, 3) == (1, 1)
    assert genotype_alleles(3, 3) == (0, 2)
    assert genotype_alleles(4, 3) == (1, 2)
    assert genotype_alleles(5, 3) == (2, 2)


def test_mendelian_compatibility_rules():
    # 00 x 00 can only produce 00.
    assert is_mendelian_compatible((0, 0), (0, 0), (0, 0))
    assert not is_mendelian_compatible((0, 0), (0, 0), (1, 1))
    assert not is_mendelian_compatible((0, 0), (0, 0), (0, 1))
    # 01 x 12 -> child 02 inherits 0 from father and 2 from mother.
    assert is_mendelian_compatible((0, 1), (1, 2), (0, 2))
    # 01 x 12 cannot produce 22 (father carries no allele 2).
    assert not is_mendelian_compatible((0, 1), (1, 2), (2, 2))
    # Homozygous child 11 from 01 x 01 is reachable.
    assert is_mendelian_compatible((0, 1), (0, 1), (1, 1))


def test_incompatibility_probability_certain_cases():
    certain = lambda index, n: [1.0 if i == index else 0.0 for i in range(n)]
    # Biallelic: father 00, mother 00, child 11 -> certainly incompatible.
    p_bad = incompatibility_probability(
        certain(0, 3), certain(0, 3), certain(2, 3), 2
    )
    assert p_bad == pytest.approx(1.0)
    # Father 01, mother 01, child 00 -> certainly compatible.
    p_ok = incompatibility_probability(
        certain(1, 3), certain(1, 3), certain(0, 3), 2
    )
    assert p_ok == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# review_trio on a hand-built cohort-style VCF
# --------------------------------------------------------------------------- #
def _trio_vcf(path: Path) -> Path:
    rows = [
        # 1. Biallelic site: both parents 0/0, child 1/1 with confident PLs
        #    -> incompatibility probability ~1.
        "chr1\t100\t.\tA\tG\t.\t.\t.\tGT:AD:PL\t"
        "0/0:20,0:0,30,250\t"
        "0/0:18,0:0,30,250\t"
        "1/1:0,17:250,30,0",
        # 2. Multi-ALT site: father 0/1, mother 1/2, child 0/2 (compatible).
        "chr1\t200\t.\tA\tG,T\t.\t.\t.\tGT:AD:PL\t"
        "0/1:10,9,0:200,0,200,200,200,200\t"
        "1/2:2,8,9:200,200,200,200,0,200\t"
        "0/2:11,1,8:200,200,200,0,200,200",
        # 3. Child GT missing -> undecidable, never counted consistent.
        "chr1\t300\t.\tC\tG\t.\t.\t.\tGT:AD:PL\t"
        "0/0:15,0:0,25,240\t"
        "0/1:9,7:0,8,190\t"
        "./.:.:.",
        # 4. Child GT present but PL entirely missing -> undecidable.
        "chr1\t400\t.\tG\tA\t.\t.\t.\tGT:AD:PL\t"
        "0/0:14,0:0,20,230\t"
        "0/0:16,0:0,18,220\t"
        "0/0:.:.",
        # 5. No-ALT site: PL still carries the triangular diploid layout.
        "chr1\t500\t.\tC\t.\t.\t.\t.\tGT:AD:PL\t"
        "0/0:20:0,30,250\t"
        "0/0:19:0,28,245\t"
        "0/0:21:0,26,240",
    ]
    return write_vcf(path, ["FA", "MO", "CH"], rows)


def test_review_trio_classifies_every_site(tmp_path):
    vcf_path = _trio_vcf(tmp_path / "cohort.vcf")
    report = review_trio(
        str(vcf_path), "FA", "MO", "CH",
        selection_version=1, threshold=0.99,
    )
    assert report.total_sites == 5
    # Sites 1, 2, 5 are decidable; 3 and 4 are undecidable.
    assert report.decidable_sites == 3
    assert report.consistent_sites == 2
    assert len(report.undecidable_sites) == 2
    reasons = " ".join(s.reason for s in report.undecidable_sites)
    assert "GT 缺失" in reasons
    assert "PL 缺项" in reasons

    assert len(report.incompatible_sites) == 1
    site = report.incompatible_sites[0]
    assert (site.chrom, site.pos) == ("chr1", 100)
    assert site.father_gt == "0/0"
    assert site.mother_gt == "0/0"
    assert site.child_gt == "1/1"
    assert site.incompatibility_probability >= 0.99

    payload = report.to_dict()
    assert payload["incompatible_count"] == 1
    assert payload["undecidable_count"] == 2
    assert payload["consistent_sites"] == 2


def test_review_trio_threshold_filters_borderline_site(tmp_path):
    # Weak PLs everywhere: parents mostly 00, child mostly 11 but with real
    # residual likelihoods, so the incompatibility mass lands between 0.5 and
    # 0.99 instead of being essentially certain.
    rows = [
        "chr1\t100\t.\tA\tG\t.\t.\t.\tGT:AD:PL\t"
        "0/0:10,1:0,3,6\t0/0:9,1:0,3,6\t1/1:1,9:6,3,0",
    ]
    vcf_path = write_vcf(tmp_path / "weak.vcf", ["FA", "MO", "CH"], rows)
    strict = review_trio(str(vcf_path), "FA", "MO", "CH", 1, 0.99)
    # The site is decidable but does not reach 0.99.
    assert strict.decidable_sites == 1
    assert strict.incompatible_sites == []
    assert strict.consistent_sites == 1
    relaxed = review_trio(str(vcf_path), "FA", "MO", "CH", 1, 0.5)
    assert len(relaxed.incompatible_sites) == 1
    assert 0.5 <= relaxed.incompatible_sites[0].incompatibility_probability < 0.99


def test_review_trio_rejects_unknown_sample(tmp_path):
    vcf_path = _trio_vcf(tmp_path / "cohort.vcf")
    with pytest.raises(PedigreeError):
        review_trio(str(vcf_path), "FA", "MO", "NOPE", 1, 0.99)


# --------------------------------------------------------------------------- #
# HTTP / workflow integration
# --------------------------------------------------------------------------- #
def _designed_batch(tmp_path: Path) -> tuple[Path, Path]:
    """Two valid files with disjoint samples (no duplicate-group resolution)."""
    file_a = write_vcf(
        tmp_path / "file_a.vcf",
        ["FA", "MO"],
        [
            # Both parents 0/0 while the child (file B) is 1/1 -> incompatible.
            "chr1\t100\t.\tA\tG\t.\t.\t.\tGT:AD:PL\t"
            "0/0:20,0:0,30,250\t0/0:18,0:0,30,250",
            # Multi-ALT site: father 0/1, mother 1/2; child 0/2 is compatible.
            "chr1\t200\t.\tA\tG,T\t.\t.\t.\tGT:AD:PL\t"
            "0/1:10,9,0:200,0,200,200,200,200\t"
            "1/2:2,8,9:200,200,200,200,0,200",
            "chr1\t300\t.\tC\tG\t.\t.\t.\tGT:AD:PL\t"
            "0/0:15,0:0,25,240\t0/1:9,7:0,8,190",
            "chr1\t400\t.\tG\tA\t.\t.\t.\tGT:AD:PL\t"
            "0/0:14,0:0,20,230\t0/0:16,0:0,18,220",
            # No-ALT site with the usual triangular PL layout.
            "chr1\t500\t.\tC\t.\t.\t.\t.\tGT:AD:PL\t"
            "0/0:20:0,30,250\t0/0:19:0,28,245",
        ],
    )
    file_b = write_vcf(
        tmp_path / "file_b.vcf",
        ["CH", "EX"],
        [
            "chr1\t100\t.\tA\tG\t.\t.\t.\tGT:AD:PL\t"
            "1/1:0,17:250,30,0\t0/0:12,0:0,22,240",
            "chr1\t200\t.\tA\tG,T\t.\t.\t.\tGT:AD:PL\t"
            "0/2:11,1,8:200,200,200,0,200,200\t"
            "0/0:14,0,0:0,30,250,30,250,250",
            # Child missing GT at chr1:300 -> undecidable for the trio.
            "chr1\t300\t.\tC\tG\t.\t.\t.\tGT:AD:PL\t"
            "./.:.:.\t0/1:8,6:0,9,180",
            # Child GT present, PL missing -> undecidable.
            "chr1\t400\t.\tG\tA\t.\t.\t.\tGT:AD:PL\t"
            "0/0:.:.\t0/0:13,0:0,21,235",
            "chr1\t500\t.\tC\t.\t.\t.\t.\tGT:AD:PL\t"
            "0/0:21:0,26,240\t0/0:17:0,29,248",
        ],
    )
    return file_a, file_b


def _upload_and_generate(client, paths: list[Path]) -> str:
    handles = [open(p, "rb") for p in paths]
    try:
        resp = client.post(
            "/api/batches",
            files=[("files", (p.name, h, "text/plain")) for p, h in zip(paths, handles)],
        )
    finally:
        for h in handles:
            h.close()
    assert resp.status_code == 200, resp.text
    batch_id = resp.json()["id"]
    gen = client.post(f"/api/batches/{batch_id}/generate")
    assert gen.status_code == 200, gen.text

    deadline = time.time() + 10
    while time.time() < deadline:
        data = client.get(f"/api/batches/{batch_id}").json()
        latest = data["latest_generation"]
        if latest and latest["status"] in ("done", "failed", "superseded"):
            assert latest["status"] == "done", latest.get("error")
            return batch_id
        time.sleep(0.05)
    raise AssertionError("generation did not finish")


def test_pedigree_requires_generation(client, tmp_path):
    file_a, file_b = _designed_batch(tmp_path)
    handles = [open(file_a, "rb"), open(file_b, "rb")]
    try:
        resp = client.post(
            "/api/batches",
            files=[
                ("files", ("file_a.vcf", handles[0], "text/plain")),
                ("files", ("file_b.vcf", handles[1], "text/plain")),
            ],
        )
    finally:
        for h in handles:
            h.close()
    batch_id = resp.json()["id"]
    review = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "FA", "mother": "MO", "child": "CH"},
    )
    assert review.status_code == 409
    assert "先生成" in review.json()["detail"]


def test_pedigree_requires_three_distinct_samples(client, tmp_path):
    batch_id = _upload_and_generate(client, list(_designed_batch(tmp_path)))
    review = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "FA", "mother": "FA", "child": "CH"},
    )
    assert review.status_code == 409
    assert "不同" in review.json()["detail"]

    review_unknown = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "FA", "mother": "MO", "child": "GHOST"},
    )
    assert review_unknown.status_code == 409
    assert "不存在" in review_unknown.json()["detail"]


def test_pedigree_full_workflow(client, tmp_path):
    batch_id = _upload_and_generate(client, list(_designed_batch(tmp_path)))
    review = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "FA", "mother": "MO", "child": "CH"},
    )
    assert review.status_code == 200, review.text
    data = review.json()
    assert data["father"] == "FA"
    assert data["mother"] == "MO"
    assert data["child"] == "CH"
    assert data["total_sites"] == 5
    assert data["decidable_sites"] == 3
    assert data["consistent_sites"] == 2
    assert data["undecidable_count"] == 2
    assert data["incompatible_count"] == 1
    site = data["incompatible_sites"][0]
    assert site["chrom"] == "chr1" and site["pos"] == 100
    assert site["father_gt"] == "0/0"
    assert site["mother_gt"] == "0/0"
    assert site["child_gt"] == "1/1"
    assert site["incompatibility_probability"] >= 0.99
    undecidable = {(s["chrom"], s["pos"]): s["reason"]
                   for s in data["undecidable_sites"]}
    assert "GT 缺失" in undecidable[("chr1", 300)]
    assert "PL 缺项" in undecidable[("chr1", 400)]


def test_pedigree_refused_against_stale_cohort_and_works_after_regenerate(client):
    from tests.conftest import DEMO_DIR

    with open(DEMO_DIR / "batch_a.vcf", "rb") as a, \
            open(DEMO_DIR / "batch_b.vcf", "rb") as b:
        resp = client.post(
            "/api/batches",
            files=[
                ("files", ("batch_a.vcf", a, "text/plain")),
                ("files", ("batch_b.vcf", b, "text/plain")),
            ],
        )
    data = resp.json()
    batch_id = data["id"]
    group = next(g for g in data["dup_groups"] if g["sample_name"] == "S2")
    a_choice = next(c["file_id"] for c in group["choices"]
                    if "batch_a" in c["file_name"])
    b_choice = next(c["file_id"] for c in group["choices"]
                    if "batch_b" in c["file_name"])
    client.put(f"/api/batches/{batch_id}/selection/{group['id']}",
               data={"file_id": a_choice})
    client.post(f"/api/batches/{batch_id}/generate")

    deadline = time.time() + 10
    while time.time() < deadline:
        latest = client.get(f"/api/batches/{batch_id}").json()["latest_generation"]
        if latest and latest["status"] == "done":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("first generation did not finish")

    ok = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "S1", "mother": "S2", "child": "S3"},
    )
    assert ok.status_code == 200, ok.text
    version_before = ok.json()["selection_version"]

    # Changing the source selection invalidates the cohort result: the review
    # endpoint must refuse to present a QC conclusion on the stale VCF.
    client.put(f"/api/batches/{batch_id}/selection/{group['id']}",
               data={"file_id": b_choice})
    stale = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "S1", "mother": "S2", "child": "S3"},
    )
    assert stale.status_code == 409
    assert "失效" in stale.json()["detail"]
    assert "重新生成" in stale.json()["detail"]

    # After regenerating under the current selection the review works again.
    client.post(f"/api/batches/{batch_id}/generate")
    deadline = time.time() + 10
    while time.time() < deadline:
        latest = client.get(f"/api/batches/{batch_id}").json()["latest_generation"]
        if latest and latest["status"] == "done":
            assert latest["selection_version"] > version_before
            break
        time.sleep(0.05)
    else:
        raise AssertionError("second generation did not finish")
    again = client.post(
        f"/api/batches/{batch_id}/pedigree",
        data={"father": "S1", "mother": "S2", "child": "S3"},
    )
    assert again.status_code == 200, again.text
