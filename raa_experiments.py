#!/usr/bin/env python3
"""
RAA Experiments (ASE Submission Artifact) — v1.0

Implements a reproducible evaluation harness for compliance mapping framed as
traceability recovery, including:

Baselines (fully reproducible):
  - tfidf : TF-IDF cosine retrieval
  - bm25  : Okapi BM25 retrieval (Robertson & Zaragoza, 2009; DOI 10.1561/1500000019)
  - lsi   : Legacy traceability baseline (LSI / truncated SVD) aligned to Marcus & Maletic (ICSE 2003)

Semantic pipeline (pluggable; optional if deps available):
  - semantic : dual-encoder retrieval + optional cross-encoder reranker (Sentence-Transformers)
             NOTE: This backend requires external dependencies and model downloads.

Evaluation protocol:
  - Stratified split by framework family: holdout test ratio, calibration ratio
  - Repeated seeds
  - Metrics: Top-1, Recall@k, MRR@k, MAP@k, nDCG@k, Precision@k, F1@k,
             Coverage, Selective Accuracy@target coverage
  - CI95 computed with Student t distribution across runs

Online Resource conventions (journal SI):
  - Exportable compliance matrix and gap report CSVs.

Usage examples:
  python raa_experiments.py --backend lsi --runs 5
  python raa_experiments.py --backend bm25 --runs 30 --holdout 0.2 --cal 0.15
  python raa_experiments.py --backend semantic --models legal=... control=... reranker=... --runs 5
  python raa_experiments.py --export-matrix --gap-report --output-dir ./output

"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# -----------------------------
# Determinism
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Control:
    control_id: int
    text: str
    regulation_id: int  # -1 for negatives
    quality: float
    match_type: str
    family: str


@dataclass(frozen=True)
class Regulation:
    regulation_id: int
    text: str
    framework: str


@dataclass(frozen=True)
class Candidate:
    control_id: int
    score: float


@dataclass(frozen=True)
class Decision:
    status: str  # accept | abstain
    confidence: float
    gap: float
    ranked: List[Candidate]


# -----------------------------
# Built-in synthetic benchmark (reconstructed from provided program)
# -----------------------------

def load_builtin_synthetic() -> Tuple[List[Regulation], List[Control], Dict[int, List[int]]]:
    regs_text = [
        # GDPR (0–7)
        "GDPR Art 32: Implement appropriate technical measures including encryption of personal data.",
        "GDPR Art 17: Establish procedures for data subject erasure requests (right to be forgotten).",
        "GDPR Art 33: Notify supervisory authority within 72 hours of becoming aware of a breach.",
        "GDPR Art 35: Conduct data protection impact assessments for high-risk processing.",
        "GDPR Art 25: Implement data protection by design and by default.",
        "GDPR Art 30: Maintain records of processing activities.",
        "GDPR Art 28: Use processors providing sufficient guarantees; ensure contracts cover security measures.",
        "GDPR Art 37: Designate a Data Protection Officer where required.",
        # NIST (8–19)
        "NIST PR.AC-1: Identities and credentials are managed for authorized devices and users.",
        "NIST PR.AC-3: Remote access is managed.",
        "NIST PR.IP-1: Baseline configuration is established and maintained.",
        "NIST PR.DS-2: Data-in-transit is protected.",
        "NIST PR.PT-1: Audit/log records are determined, documented, implemented, and reviewed.",
        "NIST DE.CM-1: The network is monitored to detect potential cybersecurity events.",
        "NIST PR.IP-4: Backups are conducted, maintained, and tested.",
        "NIST IA-2: Identification and authentication (organizational users).",
        "NIST IR-4: Incident handling.",
        "NIST SC-7: Boundary protection.",
        "NIST SC-8: Transmission confidentiality and integrity.",
        "NIST SI-4: System monitoring.",
        # HIPAA (20–29)
        "HIPAA 164.312(a): Access control for ePHI.",
        "HIPAA 164.312(b): Audit controls for systems containing ePHI.",
        "HIPAA 164.312(c): Integrity controls for ePHI.",
        "HIPAA 164.312(d): Person or entity authentication.",
        "HIPAA 164.312(e): Transmission security for ePHI.",
        "HIPAA 164.308(a)(1): Security management process (risk analysis).",
        "HIPAA 164.308(a)(5): Security awareness and training.",
        "HIPAA 164.310(d): Device and media controls (disposal, reuse).",
        "HIPAA breach response: notification workflow.",
        "HIPAA physical safeguards: facility access controls.",
        # PCI (30–39)
        "PCI DSS 1.1: Establish firewall configuration standards.",
        "PCI DSS 2.1: Change vendor-supplied defaults.",
        "PCI DSS 3.4: Render PAN unreadable wherever stored.",
        "PCI DSS 4.1: Encrypt transmission of cardholder data over open, public networks.",
        "PCI DSS 5.1: Deploy anti-malware in the cardholder environment.",
        "PCI DSS 7.1: Restrict access by business need to know.",
        "PCI DSS 8.2: Strong user authentication management.",
        "PCI DSS 10.2: Implement audit trails.",
        "PCI DSS 11.2: Run vulnerability scans regularly.",
        "PCI DSS 12.10: Implement an incident response plan.",
        # ISO (40–47)
        "ISO 27001 A.5.15: Access control policy established and reviewed.",
        "ISO 27001 A.8.16: Monitoring activities performed to detect anomalies.",
        "ISO 27001 A.8.12: Data leakage prevention procedures established.",
        "ISO 27001 A.5.23: Supplier relationship security addressed.",
        "ISO 27001 A.8.25: Secure development lifecycle implemented.",
        "ISO 27001 A.8.9: Configuration management applied.",
        "ISO 27001 A.8.28: Secure coding practices and reviews performed.",
        "ISO 27001 A.8.23: Web filtering controls implemented where needed.",
        # SOX (48–51)
        "SOX ITGC: Restrict access to financial systems by role.",
        "SOX ITGC: Maintain audit trails for financial system changes and access.",
        "SOX ITGC: Periodically review user access rights.",
        "SOX ITGC: Ensure change management approvals for financial reporting systems.",
        # SOC2 (52–57)
        "SOC 2 CC6.1: Implement logical access security.",
        "SOC 2 CC6.2: Register and authorize users prior to access.",
        "SOC 2 CC6.3: Implement access removal procedures.",
        "SOC 2 CC7.1: Detect and respond to security incidents.",
        "SOC 2 CC7.2: Monitor system components for anomalies.",
        "SOC 2 CC8.1: Authorize and implement changes to meet requirements.",
    ]

    fw = {}
    for i in range(0, 8): fw[i] = "GDPR"
    for i in range(8, 20): fw[i] = "NIST"
    for i in range(20, 30): fw[i] = "HIPAA"
    for i in range(30, 40): fw[i] = "PCI"
    for i in range(40, 48): fw[i] = "ISO"
    for i in range(48, 52): fw[i] = "SOX"
    for i in range(52, 58): fw[i] = "SOC2"

    regs = [Regulation(i, regs_text[i], fw[i]) for i in range(len(regs_text))]

    controls_raw = [
        # GDPR
        ("All databases encrypt PII at rest using AES-256 with FIPS 140-2 modules.", 0, 1.0, "perfect", "gdpr_enc"),
        ("AWS S3 buckets with personal data have SSE-S3 encryption enabled.", 0, 0.9, "good", "gdpr_enc"),
        ("Backup tapes encrypted with AES-256 before offsite transport.", 0, 0.85, "good", "gdpr_enc"),
        ("Automated data subject deletion workflow in ServiceNow with 30-day SLA.", 1, 1.0, "perfect", "gdpr_erasure"),
        ("Self-service portal for data subjects to submit erasure requests.", 1, 1.0, "perfect", "gdpr_erasure"),
        ("Breach notification procedure with 72-hour escalation documented.", 2, 1.0, "perfect", "gdpr_breach"),
        ("Security incident team activates breach assessment within 4 hours.", 2, 0.9, "good", "gdpr_breach"),
        ("DPIA template mandatory for all high-risk processing activities.", 3, 1.0, "perfect", "gdpr_dpia"),
        ("Privacy by design checklist integrated into SDLC.", 4, 1.0, "perfect", "gdpr_pbd"),
        ("Processing activity register maintained in OneTrust.", 5, 1.0, "perfect", "gdpr_ropa"),
        ("Data processing agreements executed with all third-party processors.", 6, 1.0, "perfect", "gdpr_dpa"),
        ("Data Protection Officer appointed and registered.", 7, 1.0, "perfect", "gdpr_dpo"),

        # NIST
        ("IAM lifecycle management with joiner/mover/leaver controls.", 8, 1.0, "perfect", "nist_prac1"),
        ("Privileged access managed via PAM with approvals.", 8, 0.9, "good", "nist_prac1"),
        ("VPN remote access restricted and logged.", 9, 1.0, "perfect", "nist_prac3"),
        ("Bastion host required for remote admin sessions.", 9, 0.9, "good", "nist_prac3"),
        ("Baseline configuration defined via IaC and reviewed quarterly.", 10, 1.0, "perfect", "nist_prip1"),
        ("TLS 1.2+ enforced for all service-to-service communications.", 11, 1.0, "perfect", "nist_prds2"),
        ("Centralized logging to SIEM with weekly review.", 12, 1.0, "perfect", "nist_prpt1"),
        ("IDS/IPS monitoring with alert triage for anomalies.", 13, 1.0, "perfect", "nist_decm1"),
        ("Daily encrypted backups with weekly integrity verification.", 14, 1.0, "perfect", "nist_prip4"),
        ("Backup restoration tested quarterly.", 14, 0.9, "good", "nist_prip4"),
        ("MFA required for all remote and privileged access.", 15, 1.0, "perfect", "nist_ia2"),
        ("Incident response playbook for detection and recovery.", 16, 1.0, "perfect", "nist_ir4"),
        ("Next-gen firewall with IPS at network perimeter.", 17, 1.0, "perfect", "nist_sc7"),
        ("TLS 1.3 required for all data transmission.", 18, 1.0, "perfect", "nist_sc8"),
        ("EDR deployed on all endpoints with behavioral analysis.", 19, 1.0, "perfect", "nist_si4"),

        # HIPAA
        ("EHR access restricted by role and department.", 20, 1.0, "perfect", "hipaa_access"),
        ("All ePHI access logged to SIEM with 6-year retention.", 21, 1.0, "perfect", "hipaa_audit"),
        ("Database integrity controls with checksums for ePHI.", 22, 1.0, "perfect", "hipaa_integrity"),
        ("Biometric authentication for clinical workstations.", 23, 1.0, "perfect", "hipaa_auth"),
        ("TLS 1.3 required for all ePHI transmission.", 24, 1.0, "perfect", "hipaa_trans"),
        ("Annual HIPAA risk assessment by third party.", 25, 1.0, "perfect", "hipaa_risk"),
        ("HIPAA security awareness training required annually.", 26, 1.0, "perfect", "hipaa_training"),
        ("NIST 800-88 compliant media sanitization for ePHI.", 27, 1.0, "perfect", "hipaa_media"),
        ("HIPAA breach response procedure with notification workflow.", 28, 1.0, "perfect", "hipaa_incident"),
        ("Badge access controls for data centers with ePHI.", 29, 1.0, "perfect", "hipaa_physical"),

        # PCI
        ("Firewall with documented ruleset and change control.", 30, 1.0, "perfect", "pci_firewall"),
        ("All default passwords changed before production.", 31, 1.0, "perfect", "pci_defaults"),
        ("PAN tokenized before storage.", 32, 1.0, "perfect", "pci_pan"),
        ("TLS 1.2+ required for cardholder data transmission.", 33, 1.0, "perfect", "pci_trans"),
        ("EDR deployed in cardholder environment.", 34, 1.0, "perfect", "pci_av"),
        ("RBAC enforced for cardholder environment.", 35, 1.0, "perfect", "pci_access"),
        ("MFA required for admin access to cardholder systems.", 36, 1.0, "perfect", "pci_auth"),
        ("Audit trails enabled for all CDE components.", 37, 1.0, "perfect", "pci_logs"),
        ("Quarterly vulnerability scans performed.", 38, 1.0, "perfect", "pci_scans"),
        ("Incident response plan includes payment card procedures.", 39, 1.0, "perfect", "pci_ir"),

        # ISO
        ("Access control policy reviewed annually.", 40, 1.0, "perfect", "iso_access"),
        ("Monitoring and anomaly detection alerts reviewed daily.", 41, 1.0, "perfect", "iso_monitor"),
        ("Data leakage prevention policies enforced.", 42, 1.0, "perfect", "iso_dlp"),
        ("Supplier security requirements included in contracts.", 43, 1.0, "perfect", "iso_supplier"),
        ("Secure SDLC with code review and security testing gates.", 44, 1.0, "perfect", "iso_sdlc"),
        ("Configuration baselines defined and changes reviewed.", 45, 1.0, "perfect", "iso_config"),
        ("Secure coding practices and peer review required.", 46, 1.0, "perfect", "iso_securecoding"),
        ("Web filtering applied for high-risk categories.", 47, 1.0, "perfect", "iso_web"),

        # SOX
        ("Financial system access controlled by role; quarterly recertification.", 48, 1.0, "perfect", "sox_access"),
        ("Audit logs retained for financial systems; reviewed for anomalies.", 49, 1.0, "perfect", "sox_logs"),
        ("User access reviews for financial applications performed monthly.", 50, 1.0, "perfect", "sox_review"),
        ("Change tickets require approval for financial reporting systems.", 51, 1.0, "perfect", "sox_change"),

        # SOC2
        ("Logical access security with SSO and conditional access policies.", 52, 1.0, "perfect", "soc2_cc61"),
        ("User provisioning requires approval before granting access.", 53, 1.0, "perfect", "soc2_cc62"),
        ("Automated deprovisioning within 24 hours.", 54, 1.0, "perfect", "soc2_cc63"),
        ("SIEM with correlation rules for incident detection.", 55, 1.0, "perfect", "soc2_cc71"),
        ("User behavior analytics for anomaly detection.", 56, 1.0, "perfect", "soc2_cc72"),
        ("Change advisory board reviews production changes.", 57, 1.0, "perfect", "soc2_cc81"),

        # Hard negatives / irrelevant
        ("Network traffic uses TLS (in-transit only, not at-rest).", -1, 0.0, "hard_neg", "neg_scope"),
        ("Encryption keys stored in HSM (key mgmt, not data encryption).", -1, 0.0, "hard_neg", "neg_scope"),
        ("Annual security training (training, not access control).", -1, 0.0, "hard_neg", "neg_family"),
        ("Physical badge access to building (physical, not logical).", -1, 0.0, "hard_neg", "neg_family"),
        ("Office supplies inventory system.", -1, 0.0, "neg", "neg_irrelevant"),
        ("Marketing campaign metrics tracked.", -1, 0.0, "neg", "neg_irrelevant"),
    ]

    ctrls = [Control(i, t, rid, q, mt, fam) for i, (t, rid, q, mt, fam) in enumerate(controls_raw)]

    gt: Dict[int, List[int]] = {}
    for r in regs:
        gt[r.regulation_id] = [
            c.control_id for c in ctrls
            if c.regulation_id == r.regulation_id and c.match_type in {"perfect", "good", "mapped"}
        ]
    return regs, ctrls, gt


# -----------------------------
# Splitting (stratified by framework family)
# -----------------------------

def stratified_split(
    regs: Sequence[Regulation],
    holdout_ratio: float,
    cal_ratio: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    fw_to_ids: Dict[str, List[int]] = {}
    for r in regs:
        fw_to_ids.setdefault(r.framework, []).append(r.regulation_id)

    train_ids: List[int] = []
    cal_ids: List[int] = []
    test_ids: List[int] = []

    for fw, ids in fw_to_ids.items():
        ids = list(ids)
        rng.shuffle(ids)

        n_test = max(1, int(round(len(ids) * holdout_ratio)))
        n_test = min(n_test, len(ids) - 1)

        test = ids[:n_test]
        rem = ids[n_test:]

        n_cal = max(1, int(round(len(ids) * cal_ratio)))
        n_cal = min(n_cal, max(1, len(rem) - 1))

        cal = rem[:n_cal]
        train = rem[n_cal:]

        test_ids.extend(test)
        cal_ids.extend(cal)
        train_ids.extend(train)

    return sorted(train_ids), sorted(cal_ids), sorted(test_ids)


# -----------------------------
# Tokenization + BM25
# -----------------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class BM25Index:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.docs = list(docs)
        self.tokens = [tokenize(d) for d in self.docs]
        self.N = len(self.docs)
        self.doc_lens = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens)) if self.N > 0 else 0.0

        df: Dict[str, int] = {}
        for tks in self.tokens:
            for term in set(tks):
                df[term] = df.get(term, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, dfi in df.items():
            # Robertson-Sparck Jones IDF with smoothing (common BM25 variant)
            self.idf[term] = math.log((self.N - dfi + 0.5) / (dfi + 0.5) + 1.0)

        self.tf = []
        for tks in self.tokens:
            counts: Dict[str, int] = {}
            for term in tks:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.N, dtype=float)
        if self.N == 0:
            return scores

        for term in q:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf_map in enumerate(self.tf):
                f = tf_map.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                scores[i] += idf * (f * (self.k1 + 1.0)) / denom
        return scores


# -----------------------------
# LSI baseline (legacy)
# -----------------------------

class LSIIndex:
    def __init__(
        self,
        control_docs: Sequence[str],
        fit_reg_docs: Sequence[str],
        n_components: int,
        include_regs_in_fit: bool = True,
    ):
        self.control_docs = list(control_docs)
        self.fit_reg_docs = list(fit_reg_docs)
        self.n_components = int(n_components)
        self.include_regs_in_fit = bool(include_regs_in_fit)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=0)

        self.control_latent = None
        self._fit()

    def _fit(self) -> None:
        if self.include_regs_in_fit:
            corpus = self.control_docs + self.fit_reg_docs
        else:
            corpus = self.control_docs
        X = self.vectorizer.fit_transform(corpus)
        X_latent = self.svd.fit_transform(X)
        X_latent = normalize(X_latent)
        self.control_latent = X_latent[: len(self.control_docs)]

    def score(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        q_latent = self.svd.transform(q)
        q_latent = normalize(q_latent)
        sims = (q_latent @ self.control_latent.T).reshape(-1)
        return sims


# -----------------------------
# Semantic backend interface (optional)
# -----------------------------

class SemanticBackend:
    """Pluggable interface for dual-encoder retrieval + optional cross-encoder reranking."""
    def __init__(self, controls: Sequence[str], args: argparse.Namespace):
        self.controls = list(controls)
        self.args = args

    def score(self, query: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformersBackend(SemanticBackend):
    """
    Optional backend. Requires:
      pip install sentence-transformers torch

    Model suggestions (user chooses):
      - legal encoder: e.g., nlpaueb/legal-bert-base-uncased (or other Legal-BERT variant)
      - control encoder: e.g., a SecureBERT checkpoint or a security-tuned SBERT model
      - reranker: cross-encoder for sentence pairs

    This is intentionally conservative: in many environments, model downloads require network access.
    """
    def __init__(self, controls: Sequence[str], args: argparse.Namespace):
        super().__init__(controls, args)
        from sentence_transformers import SentenceTransformer
        self.legal_model = SentenceTransformer(args.legal_model)
        self.control_model = SentenceTransformer(args.control_model)

        self.controls_emb = self.control_model.encode(
            self.controls, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        )

        self.reranker = None
        if args.reranker_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(args.reranker_model)

    def score(self, query: str) -> np.ndarray:
        q_emb = self.legal_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = (q_emb @ self.controls_emb.T).reshape(-1)

        if self.reranker is None:
            return sims

        top_k = min(len(self.controls), int(self.args.rerank_k))
        idx = np.argsort(-sims)[:top_k]
        pairs = [(query, self.controls[i]) for i in idx]
        rerank_scores = self.reranker.predict(pairs)
        rerank_scores = np.asarray(rerank_scores, dtype=float)

        out = np.full_like(sims, fill_value=-1e9, dtype=float)
        for local_i, ctrl_i in enumerate(idx):
            out[ctrl_i] = rerank_scores[local_i]
        return out


# -----------------------------
# Metrics
# -----------------------------

def rank(scores: np.ndarray, k: int) -> List[int]:
    idx = np.argsort(-scores)[:k]
    return [int(i) for i in idx]


def safe_mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else 0.0


def ndcg_at_k(gt: Sequence[int], pred: Sequence[int], k: int) -> float:
    gt_set = set(gt)
    pred_k = list(pred)[:k]
    dcg = 0.0
    for i, cid in enumerate(pred_k, start=1):
        rel = 1.0 if cid in gt_set else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(gt: Sequence[int], pred: Sequence[int], k: int) -> float:
    gt_set = set(gt)
    pred_k = list(pred)[:k]
    hit = 0
    s = 0.0
    for i, cid in enumerate(pred_k, start=1):
        if cid in gt_set:
            hit += 1
            s += hit / i
    denom = min(len(gt_set), k)
    return s / denom if denom > 0 else 0.0


def mrr_at_k(gt: Sequence[int], pred: Sequence[int], k: int) -> float:
    gt_set = set(gt)
    pred_k = list(pred)[:k]
    for i, cid in enumerate(pred_k, start=1):
        if cid in gt_set:
            return 1.0 / i
    return 0.0


def micro_precision_recall_at_k(gt_map: Dict[int, Sequence[int]], pred_map: Dict[int, Sequence[int]], reg_ids: Sequence[int], k: int) -> Tuple[float, float]:
    correct = 0
    retrieved = 0
    relevant = 0
    for rid in reg_ids:
        gt = set(gt_map[rid])
        pred = list(pred_map[rid])[:k]
        relevant += len(gt)
        retrieved += len(pred)
        correct += len(gt & set(pred))
    prec = correct / retrieved if retrieved > 0 else 0.0
    rec = correct / relevant if relevant > 0 else 0.0
    return prec, rec


def t_ci95(xs: Sequence[float]) -> Tuple[float, float]:
    xs = np.asarray(list(xs), dtype=float)
    n = xs.size
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(xs))
    if n == 1:
        return mean, 0.0
    std = float(np.std(xs, ddof=1))
    se = std / math.sqrt(n)
    try:
        import scipy.stats as st  # optional
        tcrit = float(st.t.ppf(0.975, df=n - 1))
    except Exception:
        # fallback: common t criticals; use 1.96 if unknown
        t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        tcrit = t_table.get(n, 1.96)
    return mean, tcrit * se


# -----------------------------
# Abstention calibration
# -----------------------------

def top1_conf_gap(scores: np.ndarray) -> Tuple[float, float]:
    if scores.size == 0:
        return 0.0, 0.0
    idx = np.argsort(-scores)
    conf = float(scores[idx[0]])
    gap = float(scores[idx[0]] - scores[idx[1]]) if idx.size >= 2 else float("inf")
    return conf, gap


def calibrate_thresholds(
    cal_scores: Dict[int, np.ndarray],
    cal_ids: Sequence[int],
    gt: Dict[int, List[int]],
    target_cov: float,
    gap_grid: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3),
) -> Tuple[float, float]:
    Q = len(cal_ids)
    need = int(math.ceil(target_cov * Q))

    best = None
    for gap_thr in gap_grid:
        passed = []
        for rid in cal_ids:
            conf, gap = top1_conf_gap(cal_scores[rid])
            if gap >= gap_thr:
                passed.append((rid, conf))
        if len(passed) < need:
            continue

        confs = sorted([c for _, c in passed], reverse=True)
        conf_thr = confs[need - 1]

        accepted = [rid for rid, conf in passed if conf >= conf_thr]
        if not accepted:
            continue

        sel_acc = safe_mean([1.0 if int(np.argmax(cal_scores[rid])) in set(gt[rid]) else 0.0 for rid in accepted])
        cov = len(accepted) / Q
        key = (sel_acc, -abs(cov - target_cov))
        if best is None or key > best[0]:
            best = (key, conf_thr, gap_thr)

    if best is None:
        # fallback: no gap gating; only confidence threshold
        confs = [top1_conf_gap(cal_scores[rid])[0] for rid in cal_ids]
        confs = sorted(confs, reverse=True)
        conf_thr = confs[need - 1]
        return float(conf_thr), 0.0

    _, conf_thr, gap_thr = best
    return float(conf_thr), float(gap_thr)


def decide(scores: np.ndarray, conf_thr: float, gap_thr: float) -> Decision:
    conf, gap = top1_conf_gap(scores)
    ranked_ids = np.argsort(-scores)
    ranked = [Candidate(int(i), float(scores[int(i)])) for i in ranked_ids]
    status = "accept" if (conf >= conf_thr and gap >= gap_thr) else "abstain"
    return Decision(status=status, confidence=conf, gap=gap, ranked=ranked)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_run(
    regs: Sequence[Regulation],
    controls: Sequence[Control],
    gt: Dict[int, List[int]],
    backend_name: str,
    seed: int,
    holdout_ratio: float,
    cal_ratio: float,
    top_k: int,
    target_cov: float,
    args: argparse.Namespace,
) -> Dict[str, float]:
    set_global_seed(seed)
    train_ids, cal_ids, test_ids = stratified_split(regs, holdout_ratio, cal_ratio, seed)

    control_texts = [c.text for c in controls]

    # fit backend (LSI uses train+cal regs for joint fitting to mirror doc+code corpus idea)
    if backend_name == "tfidf":
        vec = TfidfVectorizer(stop_words="english")
        X = normalize(vec.fit_transform(control_texts))

        def score_fn(q: str) -> np.ndarray:
            qv = normalize(vec.transform([q]))
            sims = (qv @ X.T).toarray().reshape(-1)
            return sims

    elif backend_name == "bm25":
        bm = BM25Index(control_texts)

        def score_fn(q: str) -> np.ndarray:
            return bm.score(q)

    elif backend_name == "lsi":
        fit_regs = [regs[rid].text for rid in (train_ids + cal_ids)]
        # tune n_components on cal (grid aligned with legacy practice of varying LSI dimensions)
        best = None
        for n_comp in args.lsi_dims:
            idx_lsi = LSIIndex(control_texts, fit_regs, n_components=int(n_comp), include_regs_in_fit=True)
            cal_scores = {rid: idx_lsi.score(regs[rid].text) for rid in cal_ids}
            conf_thr, gap_thr = calibrate_thresholds(cal_scores, cal_ids, gt, target_cov)
            accepted = [rid for rid in cal_ids if decide(cal_scores[rid], conf_thr, gap_thr).status == "accept"]
            sel_acc = safe_mean([1.0 if int(np.argmax(cal_scores[rid])) in set(gt[rid]) else 0.0 for rid in accepted]) if accepted else 0.0
            key = (sel_acc, -abs(len(accepted) / max(1, len(cal_ids)) - target_cov))
            if best is None or key > best[0]:
                best = (key, int(n_comp), conf_thr, gap_thr)

        _, best_dim, conf_thr, gap_thr = best
        idx_lsi = LSIIndex(control_texts, fit_regs, n_components=best_dim, include_regs_in_fit=True)

        def score_fn(q: str) -> np.ndarray:
            return idx_lsi.score(q)

    elif backend_name == "semantic":
        try:
            sb = SentenceTransformersBackend(control_texts, args)
        except Exception as e:
            raise RuntimeError(
                "Semantic backend unavailable. Install dependencies and provide model names:\n"
                "  pip install sentence-transformers torch\n"
                "Then run with:\n"
                "  --backend semantic --legal-model ... --control-model ... [--reranker-model ...]\n"
            ) from e

        def score_fn(q: str) -> np.ndarray:
            return sb.score(q)

    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # calibration
    cal_scores = {rid: score_fn(regs[rid].text) for rid in cal_ids}
    conf_thr, gap_thr = calibrate_thresholds(cal_scores, cal_ids, gt, target_cov)

    # test evaluation
    pred: Dict[int, List[int]] = {}
    accepted: List[int] = []

    per_query_top1 = []
    per_query_rr = []
    per_query_ap = []
    per_query_ndcg = []
    per_query_r1 = []
    per_query_r5 = []
    per_query_p5 = []

    for rid in test_ids:
        scores = score_fn(regs[rid].text)
        d = decide(scores, conf_thr, gap_thr)
        if d.status == "accept":
            accepted.append(rid)
            ranked_ids = [c.control_id for c in d.ranked[:top_k]]
            pred[rid] = ranked_ids
        else:
            pred[rid] = []

        gt_ids = gt[rid]
        gt_set = set(gt_ids)
        ranked = pred[rid]

        per_query_top1.append(1.0 if (ranked and ranked[0] in gt_set) else 0.0)
        per_query_rr.append(mrr_at_k(gt_ids, ranked, 5))
        per_query_ap.append(ap_at_k(gt_ids, ranked, 5))
        per_query_ndcg.append(ndcg_at_k(gt_ids, ranked, 5))

        per_query_r1.append(len(gt_set & set(ranked[:1])) / len(gt_set))
        per_query_r5.append(len(gt_set & set(ranked[:5])) / len(gt_set))
        per_query_p5.append((len(gt_set & set(ranked[:5])) / 5.0) if ranked else 0.0)

    top1 = safe_mean(per_query_top1)
    mrr5 = safe_mean(per_query_rr)
    map5 = safe_mean(per_query_ap)
    ndcg5 = safe_mean(per_query_ndcg)
    recall1 = safe_mean(per_query_r1)
    recall5 = safe_mean(per_query_r5)
    precision5 = safe_mean(per_query_p5)
    f1_5 = (0.0 if (precision5 + recall5) == 0.0 else (2.0 * precision5 * recall5 / (precision5 + recall5)))

    coverage = len(accepted) / max(1, len(test_ids))
    sel_acc = safe_mean([1.0 if (pred[rid] and pred[rid][0] in set(gt[rid])) else 0.0 for rid in accepted]) if accepted else 0.0

    # legacy-style micro precision/recall at cut points
    micro_p1, micro_r1 = micro_precision_recall_at_k(gt, pred, test_ids, 1)
    micro_p5, micro_r5 = micro_precision_recall_at_k(gt, pred, test_ids, 5)

    out = {
        "top1": top1,
        "mrr@5": mrr5,
        "map@5": map5,
        "ndcg@5": ndcg5,
        "recall@1": recall1,
        "recall@5": recall5,
        "precision@5": precision5,
        "f1@5": f1_5,
        "coverage": coverage,
        "sel_acc@80": sel_acc,
        "micro_precision@1": micro_p1,
        "micro_recall@1": micro_r1,
        "micro_precision@5": micro_p5,
        "micro_recall@5": micro_r5,
        "conf_thr": conf_thr,
        "gap_thr": gap_thr,
        "n_test": float(len(test_ids)),
    }
    if backend_name == "lsi":
        out["lsi_dim"] = float(best_dim)
    return out


def summarize(runs: List[Dict[str, float]], keys: Sequence[str]) -> Dict[str, str]:
    summary = {}
    for k in keys:
        mean, ci = t_ci95([r[k] for r in runs])
        summary[k] = f"{mean:.3f} ± {ci:.3f}"
    return summary


# -----------------------------
# Exports (Online Resource-oriented)
# -----------------------------

def export_matrix(
    regs: Sequence[Regulation],
    controls: Sequence[Control],
    gt: Dict[int, List[int]],
    backend_name: str,
    seed: int,
    args: argparse.Namespace,
    out_csv: str,
) -> None:
    # Simple "single run" matrix suitable for Online Resource export.
    # Uses full corpus scoring without splits; for audited runs, use evaluate_run outputs.
    control_texts = [c.text for c in controls]

    if backend_name == "tfidf":
        vec = TfidfVectorizer(stop_words="english")
        X = normalize(vec.fit_transform(control_texts))

        def score_fn(q: str) -> np.ndarray:
            qv = normalize(vec.transform([q]))
            return (qv @ X.T).toarray().reshape(-1)

    elif backend_name == "bm25":
        bm = BM25Index(control_texts)
        score_fn = bm.score

    elif backend_name == "lsi":
        idx_lsi = LSIIndex(control_texts, [r.text for r in regs], n_components=int(args.lsi_dims[0]), include_regs_in_fit=True)
        score_fn = idx_lsi.score

    else:
        raise ValueError("Matrix export currently supports tfidf/bm25/lsi only.")

    rows = []
    for r in regs:
        scores = score_fn(r.text)
        ids = rank(scores, args.top_k)
        rows.append({
            "regulation_id": r.regulation_id,
            "framework": r.framework,
            "regulation_text": r.text,
            "top1_control_id": ids[0] if ids else None,
            "top1_control_text": controls[ids[0]].text if ids else None,
            "top1_score": float(np.max(scores)) if scores.size else 0.0,
            "gt_control_ids": ";".join(map(str, gt[r.regulation_id])),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--backend", choices=["tfidf", "bm25", "lsi", "semantic"], default="tfidf")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout", type=float, default=0.20)
    p.add_argument("--cal", type=float, default=0.15)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-coverage", type=float, default=0.80)

    p.add_argument("--lsi-dims", type=int, nargs="+", default=[50, 100, 150])

    # semantic backend args (optional)
    p.add_argument("--legal-model", default="")
    p.add_argument("--control-model", default="")
    p.add_argument("--reranker-model", default="")
    p.add_argument("--rerank-k", type=int, default=20)

    # exports
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--export-matrix", action="store_true")
    p.add_argument("--gap-report", action="store_true")  # placeholder for future extension

    return p.parse_args()


def main() -> None:
    args = parse_args()
    regs, ctrls, gt = load_builtin_synthetic()

    os.makedirs(args.output_dir, exist_ok=True)

    # optional export
    if args.export_matrix:
        out_csv = os.path.join(args.output_dir, f"compliance_matrix_{args.backend}.csv")
        export_matrix(regs, ctrls, gt, args.backend, args.seed, args, out_csv)
        print(f"Wrote: {out_csv}")

    # evaluation runs
    runs = []
    for i in range(args.runs):
        seed = args.seed + i
        m = evaluate_run(
            regs, ctrls, gt,
            backend_name=args.backend,
            seed=seed,
            holdout_ratio=args.holdout,
            cal_ratio=args.cal,
            top_k=args.top_k,
            target_cov=args.target_coverage,
            args=args,
        )
        runs.append(m)
        print(
            f"Run {i+1}/{args.runs} seed={seed} | "
            f"Top1={m['top1']:.3f} MRR@5={m['mrr@5']:.3f} nDCG@5={m['ndcg@5']:.3f} "
            f"R@5={m['recall@5']:.3f} P@5={m['precision@5']:.3f} "
            f"Cov={m['coverage']:.3f} SelAcc@{int(args.target_coverage*100)}={m['sel_acc@80']:.3f} "
            f"(thr={m['conf_thr']:.3f}, gap={m['gap_thr']:.3f})"
        )

    keys = [
        "top1","mrr@5","map@5","ndcg@5","recall@1","recall@5","precision@5","f1@5",
        "coverage","sel_acc@80",
        "micro_precision@1","micro_recall@1","micro_precision@5","micro_recall@5"
    ]
    summary = summarize(runs, keys)

    print("\n=== Summary (mean ± 95% CI) ===")
    for k in keys:
        print(f"{k:>18}: {summary[k]}")

    # write summary CSV
    out_sum = os.path.join(args.output_dir, f"summary_{args.backend}.csv")
    pd.DataFrame([{"metric": k, "mean_ci95": summary[k]} for k in keys]).to_csv(out_sum, index=False)
    print(f"\nWrote: {out_sum}")


if __name__ == "__main__":
    main()
