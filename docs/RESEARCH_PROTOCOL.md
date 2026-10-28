# Research protocol version 0.1

## Research purpose

Working title: Class-Conditional Risk Control for Selective Poultry Disease
Classification with Quantized Neural Networks.

Question: at a prespecified classwise automatic-error budget, can a validation-
selected rule with predicted-class thresholds and a confidence margin retain
more automatic decisions than a global-threshold rule after independent
certification, including on INT8 models?

This is a candidate empirical contribution. The proposed rule is a heuristic
built from known ingredients. The package does not establish first-in-literature
novelty, a new theorem, Q2 status of a journal, or acceptance probability.
Before writing novelty claims, compare the algorithm to selective classification,
Learn Then Test, class-conditional conformal prediction and compression-aware
calibration papers. Keep a claim-by-source literature matrix with verified URLs.

## Relation to prior work in this project

EASR manuscript: lightweight model benchmark, imbalance treatment, TensorFlow
Lite conversion, compactness and simulated distortion robustness.
JCTA manuscript: acquisition-source shortcut learning, attribution, calibration
and source-shift mitigation.
This study focuses on the decision rule and the risk/automation trade-off.
Cite the related papers with their true publication/submission status and clearly
disclose the shared dataset. Reusing weights requires matching training IDs and
independent calibration; this package retrains models to make that separation clear.

## Dataset and task

Use the user's poultry fecal-image dataset with four existing labels. No new
clinical diagnoses or patient/animal-level claims are created. File prefixes
provide provisional source proxies only. The dataset ZIP is not bundled and
was unavailable on the final attachment attempt used for this deliverable.

Audit readability, label consistency, dimensions, SHA256 and pHash. The default
64-bit pHash Hamming threshold is 4; connected components are deduplicated to
one deterministic representative. Quarantine components with conflicting labels.
Manually inspect cleaning samples before freezing; do not assume pHash proves
the images originate from the same animal. No flock IDs are currently available.

Stratify randomly by class into 55% train, 15% validation, 15% calibration and
15% test. Source metadata are retained for descriptive subgroup reporting; source
counts are not forced in each class split. Use the same split for all models.
Every retained near-duplicate group belongs to one partition only.

## Hypotheses and endpoints

H1: post-INT8 recalibration changes the achievable automatic coverage at fixed
classwise risk budgets compared to transferring the FP32 rule unchanged.
H2: the proposed class-specific quantile-and-margin rule can improve automatic
coverage compared with a certified global rule at identical risk budgets and
familywise confidence accounting. These are hypotheses; negative results count.

H1 is a deployment-matched empirical hypothesis, not a first-in-literature
claim. El Allam and Hamlich (2026, doi:10.1016/j.bspc.2026.111217) already
compare FP32-derived versus INT8-derived Mondrian conformal calibration in an
embedded ECG application. This study differs in domain and in its simultaneous
high-probability true-class automatic-error certificates, but not in the general
idea that calibration should match deployed numerical precision.

Primary endpoints: certification status, automatic coverage, and per-class
automatic-error rates. Assess useful coverage only alongside certification status.
Secondary: disease-to-healthy automatic errors per true disease class, classwise
coverage, error among accepted predictions, ordinary four-class Macro-F1/recall,
file size and local CPU inference latency. Report all-defer outcomes prominently.

## Models and methods in the initial implementation

Two backbones: MobileNetV2 and EfficientNetB0 (not EfficientNet-Lite0).
Three fixed seeds: 17, 29, 43. Two formats: TFLite FP32 and full INT8.
The TinyCNN is solely an infrastructure smoke model by default.
Preprocessing is inside each network; external input is RGB float pixels 0..255.
Train-only class weighting and augmentation are fixed consistently.
Checkpoint selection maximizes validation Macro-F1; fine-tune upper layers with
batch normalization frozen. TFLite representative inputs come from train only.

Compared decision methods:

1. Maximum softmax >=0.90; empirical baseline without certification.
2. Temperature-scaled global threshold chosen on validation; unverified baseline.
3. Class-conditional LAC conformal sets, accepted only if singleton; marginal
   class-conditional coverage claim under exchangeability, not a PAC certificate.
4. Validation-selected global threshold followed by independent certification.
5. Proposed validation-selected predicted-class threshold vector and margin,
   followed by the SAME certification procedure and confidence allocation.
6. Proposed rule without certification as an ablation, explicitly unverified.
7. FP32 rule transferred unchanged to INT8 as a diagnostic, explicitly unverified.

The validation grid may be searched freely because the chosen candidate is fixed
before calibration is accessed. Exactly one global and one proposed candidate
per model/seed/precision receive certification. Count them all in the family.
No threshold repair, retraining, or alpha adjustment is allowed after viewing
calibration/test results under the same confirmatory claim.

Prospective post-pilot comparator decision: exact One-Sided Prediction (OSP) is
not added to version 0.1. OSP requires a distinct minimax training protocol and
primarily constrains false-positive mass entering predicted-class regions,
whereas the primary endpoint here conditions automatic error on the true class.
Calling a post-hoc threshold approximation OSP would be inaccurate. Therefore
retain the declared family of 24 certified policies, keep the implemented
class-conditional LAC baseline, and explicitly avoid superiority claims over
OSP or other trained selective-classification systems.

## Statistical and practical interpretation

Default alpha=0.10 per class and familywise delta=0.05 are prospective research
budgets, not clinical acceptability standards. The feasibility command also
shows zero-error sample requirements at alpha=0.05; it does not run a second
adaptive experiment. Any added sensitivity experiments need prospective
multiplicity accounting and must not replace unsuccessful primary results.

Clopper-Pearson bounds assume independent identically distributed error events
within each true class, conditional on the trained model and selected policy.
Deduplication does not establish flock-level independence. State that limitation.
Post-stratified source subgroup results are descriptive and uncertified by source.
No arbitrary-domain-shift guarantee, field effectiveness, true early-disease
detection, smartphone latency, or clinical safety claim is supported by this package.

Paired bootstrap intervals resample identical retained image groups across the
two policies and stratify by true class. They describe test uncertainty; they do
not replace calibration certification or establish farm-level generalization.
Seeds reuse the same images; report per-seed values and variability without
pretending they multiply the test/calibration sample size.

## Completion criteria for the initial study

Data audit and split checks passed; real TensorFlow conversion smoke passed;
all intended configurations either completed or have an explicit failure record;
policy artifacts precede test inference; all frozen hashes verified; complete
results include failures and zero-coverage methods; no claims exceed evidence.

Source-shift stress tests, corruptions, final publication figures, a fully verified
novelty review, and the manuscript are subsequent tasks, not completed features
of this starter. See IMPLEMENTATION_ROADMAP.md.
