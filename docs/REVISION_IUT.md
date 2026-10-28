# Post hoc intersection union sensitivity and exploratory procedures

These analyses were specified after the original calibration and test results
were viewed. They are sensitivity analyses. The original 24 policies, 96
Clopper Pearson bounds, 7 passing policies and 17 failed policies remain intact.

For a fixed policy j let R_jc be its automatic error probability conditional
on true class c. The policy null is H_j = union_c {R_jc > alpha_c}. A policy
passes only if every one-sided class bound is at most alpha_c. For any unsafe
policy there is a class c* with R_jc* > alpha_c. False policy certification
requires that the c* test reject. A level d test for that class has rejection
probability at most d. Consequently the all-class intersection test is level d;
the argument does not require independence between class tests or policies.
Bonferroni over the 24 fixed policies at d=0.05/24 controls the probability of
one or more falsely passing policies by 0.05.

This policy-level statement differs from simultaneous confidence coverage of
all 96 individual class bounds. The original d=0.05/96 accounting controls
that stronger simultaneous bound-coverage event. At d=0.05/24 the individual
displayed bounds must not be described as jointly 95 percent covering all 96
risks. Tests in `tests/test_revision.py` check the binomial test size and
simulate the 24-policy false-certification event. Simulations do not validate
within-class independence of the real images.

The within-run sensitivity uses four policies per backbone and seed, two rule
types times two precisions, giving d=0.05/(4*4)=0.05/16. It makes no 24-policy
family claim. The unadjusted sensitivity d=0.05 has no familywise claim.
The different sensitivity scenarios are not simultaneously corrected or new
confirmatory results. No successful sensitivity can replace an original failure.

## Separate exploratory comparator family

There are 12 backbone seed precision cases, six scalar confidence scores, and
two high-probability procedures, giving 144 exploratory procedures. Results
are shown at procedure delta=0.05 and delta=0.05/144. The validation-selected
CP pipeline uses class delta=procedure_delta/4 to match its simultaneous-bound
accounting. Fixed-sequence LTT uses class delta=procedure_delta and combines
the four binomial p-values by their maximum, an intersection union test. The
threshold sequence consists of 50 validation-score quantiles, deduplicated,
ordered strict to permissive. Testing stops at the first nonrejection. A
fixed sequence needs no further factor for its length: any false rejection
requires rejection of the first true null. Scores and grids are fixed using
validation, not calibration or test. As all analyses are post hoc, conditional
mathematical guarantees do not establish a new prospective study claim.

The validation-selected CP pipeline searches these quantiles in the reverse,
permissive-to-strict order, preserving the original global search convention
that the first candidate wins equal validation coverage.

## Classwise conformal risk control

For each class the selected threshold is the least conservative threshold
meeting (n_c*Rhat_c+1)/(n_c+1)<=alpha. Loss is the indicator of an accepted
incorrect decision, decreasing with the score threshold. Acceptance uses >=;
the implementation uses `nextafter(error_score,+infinity)` to exclude every
tied boundary error. If there are too few class observations for the +1
correction, the class threshold is infinity, a structural all-defer rule.
The deployed scalar threshold is the maximum of the four class thresholds.
Monotonicity means its risk for class c is no greater than that of the
class-c threshold, hence the classwise expected-risk statement is preserved.
This is an expectation guarantee under exchangeability, not a high-probability
bound or simultaneous 95 percent certification event. No delta is assigned.

## Ranking convention

DOCTOR uses the negative unperturbed D_alpha odds: -g/(1-g), with
g=1-sum_c p_c squared, so larger scores mean greater confidence. This is
monotone in sum_c p_c squared. Energy confidence is logsumexp of the saved
logits at T=1, including dequantized INT8 logits. Only the MSP temperature
score uses the original validation-fitted temperature; the other scalar
scores use raw probabilities or logits. AURC is the discrete average of
selective errors at ranks 1 through N. Tied scores use the exact expectation
under uniformly random ordering within each tie block. E-AURC subtracts the
discrete oracle on the same predictions. Curves contain attainable >= score
thresholds at complete tie blocks, including the all-defer endpoint.

Primary sources are listed in `docs/REVISION_VERIFIED_SOURCES.md`; this
document does not supply unverified bibliographic metadata.
