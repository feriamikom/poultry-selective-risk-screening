# Statistical contract

Let Y be the true dataset label, h(X) the classifier label, and A(X) the indicator
that the decision is accepted. The primary risk for each class c is

    R_c = P(A(X)=1 and h(X) != c | Y=c).

It is the probability of an AUTOMATIC ERROR among ALL examples of class c.
It is not P(error | accepted, Y=c), and is not clinical sensitivity.
The automatic disease-to-healthy risk is a subset of R_c for disease classes,
and is reported separately. Correct automatic fraction is also distinct from
ordinary recall because deferred examples have not been classified correctly.

## Selection and certification

1. Train and select weights using train/validation only.
2. Fit temperature on validation only. Search validation policy candidates.
3. Freeze one global and one proposed policy per backbone/seed/precision.
   The implementation writes `selected_policies.json` and `selection_freeze.json`
   from validation predictions before producing calibration predictions. These
   hashes are verified on reuse; certification does not refit temperature or
   search thresholds. `policy_freeze.json` then binds the certificates and final
   deployment policies before test inference.
4. On independent calibration data, count n_c examples and k_c automatic errors.
5. For M certified policies and C=4 classes, use delta'=delta/(M*C).
6. U_c = BetaInverseCDF(1-delta'; k_c+1, n_c-k_c) when k_c<n_c;
   use U_c=1 if n_c=0 or k_c=n_c.
7. Certify a nontrivial policy only if every U_c<=alpha_c. Bonferroni gives
   simultaneous coverage of all bounds in the declared family under the iid
   binomial assumptions. Policies may share calibration data; independence
   between policies is not required for the union bound.
8. If a policy fails, the deployed version defers all predictions, and the
   original failed certificate remains in the record. A structurally all-defer
   rule has zero automatic error by construction but is marked TRIVIAL_ALL_DEFER.

M = 2 * number_of_backbones * number_of_seeds * number_of_formats, declared from
the full configuration even if some runs fail or are not executed. Adding a
new method, seed or architecture after seeing results is not covered by the old M.

At k=0, U=1-(delta')**(1/n), not zero.
The minimum n to certify with zero errors is ceil(log(delta')/log(1-alpha)).
For the full default family M=24, delta'=0.05/96. This requires 72 independent
calibration images per class at alpha=0.10, and 148 at alpha=0.05. Actual counts
come from the new manifest; historic Newcastle test n=84 is not calibration n.

## Proposed policy in version 0.1

For each quantile q in a prespecified grid, estimate a score threshold t_c from
the validation examples both predicted and labeled c. If none exist set t_c>1.
Accept a sample if its highest calibrated probability >=t_predictedclass and
its top-two margin >=m. Select (q,m) with maximum validation automatic coverage
subject to per-class empirical automatic error <=0.5*alpha_c. Ties follow grid
order. The factor 0.5 is a prespecified tuning heuristic, not a confidence bound.
Certification is performed on the selected rule only; changing it after the
calibration result is disallowed. The model never receives the true test class
to choose an acceptance threshold.

This is an implementable baseline candidate, not a verified new statistical
method. Compare directly to existing class-adaptive selective-prediction methods
before claiming methodological novelty.

## Class-conditional conformal baseline

For each class c, use calibration scores 1-p_c(X) for examples with Y=c.
Select the ceil((n_c+1)*(1-alpha_c))-th order statistic. If the order exceeds n_c,
set the threshold to 1 so the label cannot be excluded. Include every class
whose score meets its own threshold; accept only singleton prediction sets.
This gives a marginal class-conditional set-coverage statement under classwise
exchangeability. It does not certify with probability 1-delta that a fixed
conditional risk is below alpha. Keep those guarantee types separate in tables.

## Boundaries

- iid/exchangeability assumptions are substantive, not verified by a unit test.
- Unknown flock/bird clustering and source changes can violate them.
- Temperature scaling must not be fitted on the certification set.
- INT8 representative conversion inputs must not use calibration/test images.
- A narrower test confidence interval does not repair a failed certificate.
- Reporting a risk budget is not a claim that the budget is clinically acceptable.
- Do not pool clean and corrupted versions as independent samples.
- Failure to certify can mean insufficient evidence, not necessarily high true risk.

Foundational references:
Learn Then Test: https://arxiv.org/abs/2110.01052
Conformal Risk Control: https://arxiv.org/abs/2208.02814
Conformal prediction introduction: https://arxiv.org/abs/2107.07511
