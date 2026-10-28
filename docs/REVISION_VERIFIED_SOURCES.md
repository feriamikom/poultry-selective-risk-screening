# Primary sources for the added confidence scores and dataset records

Accessed 2026-10-27. These entries verify the added score definitions and dataset
metadata; they do not establish novelty or validate this archive's image labels.

- DOCTOR: F. Granese, M. Romanelli, D. Gorla, C. Palamidessi, and P. Piantanida,
  "DOCTOR: A Simple Method for Detecting Misclassification Errors", Advances in
  Neural Information Processing Systems, Vol.34, 2021.
  https://papers.neurips.cc/paper_files/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html
  Full primary text: https://arxiv.org/html/2106.02395v2, Eqs. (11) and (14).
  The unperturbed totally black-box D_alpha rejects when Gini exceeds gamma
  times the sum of squared probabilities. The implemented confidence is the
  negative ratio Gini/(1-Gini), a monotone equivalent of negative Gini.
- AURC and excess AURC: Y. Geifman, G. Uziel, and R. El-Yaniv,
  "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers", International
  Conference on Learning Representations, 2019.
  https://arxiv.org/abs/1805.08206 (accepted venue stated by authors);
  full text https://arxiv.org/html/1805.08206v4, Section 3.
  The implementation uses a discrete rank average and its discrete oracle, with exact
  expected random ordering inside tied scores; this convention is explicit.
- Negative energy: W. Liu, X. Wang, J. D. Owens, and Y. Li,
  "Energy-based Out-of-distribution Detection", Advances in Neural Information
  Processing Systems, Vol.33, 2020.
  https://papers.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
  Paper Eq. (4) and Section 3.1 use negative energy as higher-is-in-distribution
  confidence. The implementation fixes T = 1 and uses stored dequantized logits. This is a
  confidence ranking comparison, not a new out-of-distribution experiment.
- Learn then Test: https://arxiv.org/abs/2110.01052. The authors' primary record
  verifies the risk-control testing framework. It is cited as reference [8] of
  the manuscript, with its published journal metadata.
- Conformal Risk Control: https://arxiv.org/abs/2208.02814. Controls expected
  monotone loss; its guarantee is distinct from high-probability risk screening.
  It is cited as reference [23] of the manuscript.
- Field collection: D. Machuve, E. Nwankwo, N. Mduma, H. Mbelwa, E. Maguo, and
  C. Munisi, "Machine Learning Dataset for Poultry Diseases Diagnostics", Zenodo,
  2021, doi:10.5281/zenodo.4628934.
  https://zenodo.org/records/4628934. Record reports 6,812 labelled images.
- PCR collection: D. Machuve, E. Nwankwo, E. Lyimo, E. Maguo, and C. Munisi,
  "Machine Learning Dataset for Poultry Diseases Diagnostics - PCR annotated",
  Zenodo, 2021, doi:10.5281/zenodo.5801834.
  https://zenodo.org/records/5801834. Record reports 1,255 labelled images.
  Both records have a blank displayed License field on access. Their sum matches
  the local archive count; this does not verify image-by-image diagnostic linkage.

No page range, conference location, DOI, or author field was inferred where the
primary record does not state it; the added references cite the verified venue.
