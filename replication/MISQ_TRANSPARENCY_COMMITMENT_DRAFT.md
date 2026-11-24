# MIS Quarterly Research Transparency Commitment Form — Ready-to-Use Draft

Use this text when completing `templates/MISQ_Research_Transparency_Commitment.pdf` for ScholarOne Step 2 ("Miscellaneous").

---

## 1. Form Fields

- **Title of Manuscript:**
  `Learning to Pay Attention: Unsupervised Modeling of Attentive and Inattentive Survey Respondents`
- **Have you reviewed the Research Transparency guidelines?**
  `[x]` Check the box.
- **Which option to select:**
  - **Option 2 (Recommended if providing an anonymous review URL such as an OSF view-only link now):**
    *"Transparency materials will be required for this paper. The author(s) agree to provide materials that accord with the guidelines."*
  - **Option 3 (Recommended if withholding the public Zenodo DOI until minor revision / acceptance to preserve double-anonymous review — aligns directly with MISQ's Burton-Jones et al., 2021 procedural workflow):**
    *"Transparency materials will be required for this paper. The author(s) agree to provide materials; however, the research involves a genre not covered in detail in the guidelines."* *(or Option 2 with the text below noting the permanent Zenodo DOI will be published upon conditional acceptance / minor revision).*

---

## 2. Copy-Paste Text for "Materials / Alternative Plan" Box

### Variant A — Recommended (Zenodo permanent archive published upon minor revision / acceptance; preserves double-anonymous review)

> To preserve double-anonymous peer review (MISQ Policy §2.1), public links to the authors' repositories and live web application are withheld during initial review. A complete, self-contained replication and research-transparency package has already been assembled and verified (SHA-256 manifest, automated Python syntax checks, and zero-sorry Lean 4 proof compilation) and will be published to Zenodo with a permanent, citable DOI upon minor revision / conditional acceptance for review by the Senior Editor and Research Transparency Editor.
>
> The Zenodo archive will include:
> 1. **Software and replication code (MIT License):** the unified Python package implementing closed-form Chow-Liu dependence trees, Percentile-Loss and Linear autoencoders, six psychometric careless-responding baselines, the Section 4.2 subgroup-fairness audit, and all experiment, simulation, and figure-generation scripts, alongside a publicly hosted companion web application for label-free CSV survey screening (URL released upon acceptance).
> 2. **Data provenance and score caches (MISQ Policy §4.5):** the public-domain 2017 CDC Youth Risk Behavior Survey dataset (17 U.S.C. § 105), precomputed anonymous per-respondent detector score caches and selected hyperparameter configurations for exact reproduction of all reported results, and canonical repository DOIs/URLs and preprocessing scripts for the eight third-party public surveys (whose raw CSVs remain governed by their original repository licenses).
> 3. **Machine-checked formalization (MIT License):** a zero-dependency Lean 4 (v4.35.0) proof package (`lean/`, zero `sorry` placeholders, zero custom axioms) independently verifying the paper's finite-support probability propositions, ordered-field bounds, factor-loading identities, sharpness counterexamples, and cost-benefit screening rules (analytical proofs of Propositions 1–2, including continuous-population extensions, appear in Appendices A–B of the 55-page manuscript).
> 4. **Technical reference documentation (CC-BY-4.0):** verified aggregate result summaries, the 242-entry manuscript number-verification ledger, and unreviewed technical documentation carrying MISQ's Research Transparency disclaimer.

### Variant B — If you also upload `inattentiveness-screening-replication-blind-v1.0.0.zip` to an anonymous OSF view-only project for Round 1 reviewers

> An anonymized peer-review copy of our replication package (code, public-domain CDC 2017 YRBS data, verified result summaries, zero-dependency Lean 4 formalization, dataset provenance guide, and unreviewed technical documentation) is available at: `<INSERT_ANONYMOUS_OSF_VIEW_ONLY_URL>`. Upon minor revision / conditional acceptance, the full de-anonymized archive (including precomputed anonymous per-respondent score caches and the companion web application URL) will be published to Zenodo under a permanent, citable DOI for verification by the Senior Editor and Research Transparency Editor.
