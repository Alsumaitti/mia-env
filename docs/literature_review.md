# Literature Review: Membership Inference Attacks and Defenses in Machine Learning

**Author:** Osamah Alsumaitti
**Course:** Independent Study — M.Sc. in Cybersecurity
**Date:** 2026-04-07
**Scope:** AI security and data privacy, with emphasis on membership inference attacks (MIAs) on supervised classifiers and large language models, together with the defenses proposed against them.

## 1. Introduction

Membership inference attacks (MIAs) ask a deceptively simple question: given a trained machine learning model and a candidate data record, can an adversary decide whether that record was part of the model's training set? A positive answer has direct privacy implications. If the training corpus consisted of hospital discharge records, financial transactions, or private messages, then leaking membership reveals sensitive facts about identifiable individuals, and in some jurisdictions qualifies as a breach of data protection law. Since Shokri et al. introduced the formal attack setting in 2017, MIAs have become the dominant empirical measuring stick for the privacy leakage of modern ML systems, and they now bridge two once-separate research communities — applied cryptography and privacy, and mainstream deep learning.

This review synthesizes fourteen seed papers that together trace the arc of the field from its foundational black-box attack formulation, through white-box and label-only generalizations, to principled re-evaluations of attack metrics and the newest attacks against large language models. It also covers the main defensive families — differentially private training, adversarial regularization, and output-perturbation heuristics — and the widely cited 2022 survey by Hu et al. The review was prepared as part of an independent study in AI security and data privacy, and is intended to support the methodology and threat-model sections of a subsequent master's thesis.

## 2. Methodology of the Review

Each of the fourteen core papers was selected because it is either widely cited as a foundation of the MIA literature, introduces a materially new attack or defense, or reframes how the community evaluates privacy leakage. For each paper, the arXiv abstract page was fetched directly (URLs are recorded in each entry under "Source verified via") and cross-checked against the author list and venue. Abstracts were paraphrased; only short phrases of under fifteen words are quoted verbatim. Where an abstract did not describe specific quantitative results, those fields are left deliberately sparse rather than filled with numbers that could not be verified from the fetched source.

## 3. Per-Paper Summaries

### 3.1 Shokri et al. (2017) — Membership Inference Attacks Against Machine Learning Models
- **Venue / Year:** IEEE Symposium on Security and Privacy, 2017.
- **Problem:** Do black-box ML models leak whether a particular record was part of their training data?
- **Method:** Train "shadow models" that imitate the target model's behavior on data drawn from a similar distribution, then train a binary "attack model" on the shadows' outputs to classify records as members or non-members based on the target model's prediction vector.
- **Datasets:** The abstract highlights realistic classification tasks including a hospital discharge dataset, and evaluates attacks against commercial ML-as-a-service offerings from Google and Amazon.
- **Key results:** Establishes that commercially deployed models are vulnerable to membership inference using only black-box query access; investigates which factors drive leakage and discusses initial mitigations.
- **Limitations / caveats:** The attack pipeline assumes the adversary can train many shadow models and obtain data drawn from a similar distribution as the target's training set — assumptions later papers explicitly relax.
- **Source verified via:** https://arxiv.org/abs/1610.05820

### 3.2 Salem et al. (2019) — ML-Leaks
- **Venue / Year:** NDSS 2019.
- **Problem:** Shokri et al.'s attack required many shadow models, knowledge of the target architecture, and similarly distributed data — are all of these assumptions necessary?
- **Method:** Progressively relaxes each assumption and proposes model- and data-independent attack variants, evaluated across a range of datasets. Also proposes defensive mechanisms intended to preserve model utility.
- **Datasets:** Eight diverse datasets (the abstract states the number but not a full enumeration).
- **Key results:** MIAs are more broadly applicable and cheaper to mount than originally thought; a single shadow model (and even no shadow model) can suffice. The paper concludes that membership inference is a more serious threat to ML services than Shokri et al. had demonstrated.
- **Limitations / caveats:** The proposed defenses are heuristic and, as later work would show, may not withstand adaptive attackers.
- **Source verified via:** https://arxiv.org/abs/1806.01246

### 3.3 Nasr, Shokri, and Houmansadr (2019) — Comprehensive Privacy Analysis of Deep Learning
- **Venue / Year:** IEEE Symposium on Security and Privacy, 2019.
- **Problem:** Extend MIAs from black-box to white-box settings, covering both centralized and federated training.
- **Method:** Introduces white-box attacks that exploit the internal parameters and, crucially, the per-step gradient updates produced by stochastic gradient descent. Considers both passive adversaries (observing updates) and active adversaries (influencing them, as participants in federated learning can do).
- **Datasets:** CIFAR and other standard vision benchmarks are referenced as target models in the abstract.
- **Key results:** Naive white-box extensions of black-box attacks are not more powerful; the authors' SGD-aware white-box attack is. Even well-generalized models remain vulnerable, and active adversarial participants in federated learning can successfully infer members of other participants' data.
- **Limitations / caveats:** The strongest attacks require observation of gradient updates over multiple rounds, which is only realistic in federated or collaborative settings.
- **Source verified via:** https://arxiv.org/abs/1812.00910

### 3.4 Yeom et al. (2018) — Privacy Risk in Machine Learning
- **Venue / Year:** IEEE Computer Security Foundations Symposium (CSF), 2018.
- **Problem:** Formalize the intuitive link between overfitting and privacy leakage for both membership and attribute inference.
- **Method:** Theoretical analysis coupled with empirical evaluation. Introduces a simple loss-threshold MIA in which records with unusually low training-time loss are flagged as members.
- **Datasets:** Standard tabular and image benchmarks are used in the empirical evaluation.
- **Key results:** Overfitting is sufficient for membership inference to succeed, but not strictly necessary — other structural properties of the learned function can also leak membership. Attribute inference attacks share a deep connection with MIAs, enabling new attack strategies.
- **Limitations / caveats:** The proposed thresholding attack is simple; more sophisticated attacks later outperform it, particularly at the low false-positive-rate regime.
- **Source verified via:** https://arxiv.org/abs/1709.01604

### 3.5 Long et al. (2018) — Understanding Membership Inferences on Well-Generalized Learning Models
- **Venue / Year:** arXiv preprint, 2018 (widely cited in the MIA literature).
- **Problem:** Do models that generalize well still leak training membership?
- **Method:** Proposes a "generalized" MIA that targets vulnerable individual records rather than attempting uniform attack performance across the dataset. Records are identified indirectly, by querying related records and observing the model's differential behavior.
- **Datasets:** Several standard classification benchmarks.
- **Key results:** Well-generalized models remain vulnerable to record-level membership inference; the vulnerability arises from the unique influence particular training instances exert on the learned function. Standard generalization techniques alone do not suffice as a defense.
- **Limitations / caveats:** Focuses on identifying a subset of vulnerable records rather than on aggregate attack accuracy, which means the threat model differs subtly from Shokri et al.'s.
- **Source verified via:** https://arxiv.org/abs/1802.04889

### 3.6 Carlini et al. (2022) — Membership Inference Attacks From First Principles
- **Venue / Year:** IEEE Symposium on Security and Privacy, 2022.
- **Problem:** The field had been reporting MIA strength using balanced accuracy or AUC, which can hide the fact that an attack with "high accuracy" may still be useless against any individual user. The paper reframes evaluation.
- **Method:** Argues that MIAs should be evaluated by true-positive rate at low false-positive rate (e.g., below 0.1% FPR), because privacy harm stems from confidently identifying specific members. Introduces the Likelihood Ratio Attack (LiRA), which calibrates per-example statistics using shadow models.
- **Datasets:** Standard image classification benchmarks, including CIFAR-style evaluations.
- **Key results:** The abstract reports that LiRA is roughly an order of magnitude more powerful at low false-positive rates than prior attacks, while also dominating them on conventional aggregate metrics.
- **Limitations / caveats:** LiRA requires training many shadow models, making it computationally expensive for large target models; the attack is strongest when the adversary can approximate the target's training distribution.
- **Source verified via:** https://arxiv.org/abs/2112.03570

### 3.7 Choquette-Choo et al. (2021) — Label-Only Membership Inference Attacks
- **Venue / Year:** ICML 2021.
- **Problem:** Can MIAs succeed when the model only returns hard labels rather than confidence scores, thereby defeating "confidence-masking" defenses?
- **Method:** Assess label robustness under input perturbations — including standard data augmentations and adversarial examples — on the intuition that training points tend to be more robust to small perturbations than non-training points.
- **Datasets:** Standard image classification benchmarks.
- **Key results:** Label-only attacks perform on par with prior confidence-based attacks. This implies that confidence masking is not a viable defense. The authors further report that only differential privacy training and strong L2 regularization meaningfully resist their attacks, even when the DP guarantee is numerically loose.
- **Limitations / caveats:** The attack requires multiple queries per candidate record (to measure robustness under perturbations), which an API rate limiter could partially mitigate.
- **Source verified via:** https://arxiv.org/abs/2007.14321

### 3.8 Abadi et al. (2016) — Deep Learning with Differential Privacy
- **Venue / Year:** ACM CCS 2016.
- **Problem:** How can deep neural networks be trained with a meaningful differential privacy guarantee while remaining practically useful?
- **Method:** Introduces DP-SGD — differentially private stochastic gradient descent — in which per-example gradients are clipped to a fixed norm and Gaussian noise is added before the update. The authors also develop the "moments accountant," a tighter way to track cumulative privacy loss across training steps.
- **Datasets:** The abstract emphasizes non-convex deep learning under "a modest privacy budget"; the paper's experiments include standard image benchmarks.
- **Key results:** Establishes DP-SGD as a practical route to training deep models with differential privacy guarantees and introduces a privacy accountant that gives tighter bounds than standard composition theorems.
- **Limitations / caveats:** DP-SGD imposes a non-trivial accuracy cost, particularly on harder datasets; subsequent work (including Papernot et al., 2021) has focused on narrowing that gap through architectural and training choices.
- **Source verified via:** https://arxiv.org/abs/1607.00133

### 3.9 Papernot et al. (2021) — Tempered Sigmoid Activations for Deep Learning with Differential Privacy
- **Venue / Year:** AAAI 2021.
- **Problem:** The accuracy cost of DP-SGD is widely assumed to be fundamental. Is it actually tied to architectural choices made for non-private training?
- **Method:** Argues that architectures should be designed for DP from the outset. Identifies activation functions as a central factor because they control per-example gradient magnitudes that drive the clipping step in DP-SGD. Proposes a family of bounded "tempered sigmoid" activations as a drop-in replacement for ReLU.
- **Datasets:** MNIST, Fashion-MNIST, and CIFAR-10.
- **Key results:** Tempered sigmoids consistently outperform ReLU under DP-SGD and yield state-of-the-art DP accuracy on the three benchmarks without changing the underlying learning procedure.
- **Limitations / caveats:** Gains are demonstrated on small-to-medium vision benchmarks; scaling to large language models or very deep networks is outside the paper's scope.
- **Source verified via:** https://arxiv.org/abs/2007.14191

### 3.10 Nasr, Shokri, and Houmansadr (2018) — Machine Learning with Membership Privacy using Adversarial Regularization
- **Venue / Year:** ACM CCS 2018.
- **Problem:** Can membership privacy be obtained without the accuracy cost associated with differential privacy?
- **Method:** Formulates training as a min-max game between the classifier and an inference adversary. The inference adversary tries to distinguish training points from non-training points based on the classifier's behavior, and the classifier is regularized to make these indistinguishable.
- **Datasets:** Several standard classification benchmarks, with membership inference attacks applied before and after defense.
- **Key results:** Reduces the success of membership inference "close to random guess" on the benchmarks tested, while reporting only negligible loss in classification accuracy.
- **Limitations / caveats:** The guarantee is empirical, not formal: resistance is measured against a specific class of attacks rather than proven against arbitrary adversaries, and later first-principles attacks (Carlini et al., 2022) have shown that empirical defenses can appear strong against weak attacks while remaining leaky at low FPR.
- **Source verified via:** https://arxiv.org/abs/1807.05852

### 3.11 Jia et al. (2019) — MemGuard
- **Venue / Year:** ACM CCS 2019.
- **Problem:** Existing defenses (DP training, adversarial regularization) modify training, which is costly and may not be an option for deployed models. Can the defender act purely on the model's output?
- **Method:** Treats the attacker's classifier as a standard ML model, and perturbs each released confidence vector into an adversarial example against that classifier. The perturbation is constrained to preserve the argmax label — and thus the model's predicted class — while flipping the membership prediction.
- **Datasets:** Three standard benchmarks.
- **Key results:** Offers the first defense claiming formal utility-loss guarantees against black-box membership inference and achieves better privacy-utility trade-offs than prior DP-based baselines within the paper's threat model.
- **Limitations / caveats:** Because the defense is itself a form of adversarial-example perturbation against a specific attacker, it has become a paradigmatic example of a heuristic defense that may not hold up against adaptive attackers aware of the defense's mechanics.
- **Source verified via:** https://arxiv.org/abs/1909.10594

### 3.12 Hu et al. (2022) — Membership Inference Attacks on Machine Learning: A Survey
- **Venue / Year:** ACM Computing Surveys, 54(11s), 2022 (arXiv:2103.07853).
- **Problem:** By 2021 the MIA literature had fragmented across classifiers, generative models, federated learning, and language models; a unified taxonomy was needed.
- **Method:** Survey of the attack and defense literature to date, with taxonomies for threat models, attack strategies, evaluation metrics, and defenses. The paper also provides a clinical-records example to motivate why membership leakage is a direct privacy breach.
- **Key contribution:** A consolidated taxonomy and reading list covering MIAs on both classification and generative models, along with an analysis of open research directions.
- **Limitations / caveats:** As a 2021–2022 survey, it predates the first-principles re-evaluation of MIA metrics (Carlini et al., 2022) and the most recent wave of attacks on large language models.
- **Source verified via:** https://arxiv.org/abs/2103.07853

### 3.13 Carlini et al. (2021) — Extracting Training Data from Large Language Models
- **Venue / Year:** USENIX Security 2021.
- **Problem:** Large language models are trained on huge web-scraped corpora. Do they memorize individual training sequences, and can those sequences be recovered by an external querier?
- **Method:** Proposes a training-data extraction attack against GPT-2. Candidate sequences are generated by sampling from the model and then ranked using membership inference–style scoring (for example, comparing the target model's perplexity with that of a reference model).
- **Datasets:** GPT-2 and its underlying web-scale training corpus.
- **Key results:** Recovers hundreds of verbatim training sequences, including personally identifiable information (names, phone numbers, email addresses), code, IRC conversations, and 128-bit UUIDs, even for sequences that appear only once in training. Larger model variants are more vulnerable than smaller ones.
- **Limitations / caveats:** The attack targets verbatim memorization rather than the broader statistical leakage that motivates classical MIAs, and evaluation is restricted to GPT-2.
- **Source verified via:** https://arxiv.org/abs/2012.07805

### 3.14 Mattern et al. (2023) — Membership Inference Attacks against Language Models via Neighbourhood Comparison
- **Venue / Year:** Findings of ACL 2023.
- **Problem:** Prior language-model MIAs either threshold on the model's own loss (which produces many false positives) or use a reference-model baseline (which requires unrealistically close access to the target's training distribution).
- **Method:** Proposes "neighbourhood attacks" that compare the target model's score for a candidate sample against its scores on synthetically generated neighbouring texts (obtained, for example, by masked-language-model rewriting). No reference model trained on a similar corpus is required.
- **Datasets:** Standard language-model benchmarks.
- **Key results:** The neighbourhood approach is competitive with reference-based attacks that assume perfect knowledge of the training distribution, while strictly outperforming reference-free baselines and reference-based attacks that rely on imperfect reference corpora.
- **Limitations / caveats:** Neighbour generation depends on having a reasonable rewriting model; the attack also inherits the general difficulty of performing MIA on models trained on massive, heterogeneous text corpora where any single example has limited influence.
- **Source verified via:** https://arxiv.org/abs/2305.18462

## 4. Thematic Synthesis

### 4.1 Foundational attacks
Shokri et al. (2017), Yeom et al. (2018), and Long et al. (2018) form the foundation of the field. Shokri et al. defined the threat model that still organizes the area: a black-box adversary with query access attempts to distinguish training from non-training records, and the attack is operationalized via shadow models that imitate the target's behavior. Yeom et al. then asked *why* these attacks worked and tied their success directly to overfitting, demonstrating that a simple per-example loss threshold is often enough to mount a meaningful attack. Long et al. complicated this narrative by showing that even well-generalized models leak membership for a subset of unusually "influential" training records. Taken together, the three papers establish a consistent picture: overfitting is a sufficient driver of leakage, but not the only one, and aggregate generalization metrics alone cannot be used as a proxy for privacy.

### 4.2 Stronger attacks and relaxed assumptions
The next wave of attacks attacked the two weakest points of the foundational pipeline: its assumptions (Salem et al., 2019; Choquette-Choo et al., 2021) and its evaluation methodology (Carlini et al., 2022), while also extending it to richer threat models (Nasr, Shokri, and Houmansadr, 2019). Salem et al. showed that shadow-model abundance, architectural knowledge, and distributional similarity are not prerequisites — making MIAs a realistic off-the-shelf threat. Nasr et al. pushed in the opposite direction by adding information: their white-box and federated-learning attacks exploit gradient updates rather than output vectors, and reveal that well-generalized CIFAR models remain exposed when the adversary sees internal state. Choquette-Choo et al. removed information instead, demonstrating that hard-label outputs alone suffice. The four attacks therefore bracket the threat model from both ends.

Carlini et al. (2022) then reframed the field's evaluation practice. Rather than asking for balanced accuracy, the paper argued that privacy risk is meaningful only at very low false-positive rates — because a defender does not care whether an attacker can flag half the training set if it also flags half the test set, but does care very much if the attacker can identify even a small number of members with high confidence. Their LiRA attack substantially outperforms prior work under this metric. This is not merely a cosmetic shift: several earlier defenses that looked strong on aggregate metrics are implicitly re-evaluated by this critique, and open questions about the strength of Nasr et al. (2018) and Jia et al. (2019) follow directly from it.

### 4.3 Defenses
Defenses divide into two families. The principled family begins with Abadi et al. (2016), whose DP-SGD provides a formal (epsilon, delta) guarantee by clipping per-example gradients and adding calibrated Gaussian noise. Papernot et al. (2021) refine DP-SGD by arguing that the architectural choices inherited from non-private training — particularly ReLU activations — interact badly with gradient clipping, and that bounded "tempered sigmoid" activations recover much of the accuracy loss. Together they represent the defensive path with the strongest theoretical standing; Choquette-Choo et al.'s finding that only DP training and strong L2 regularization resist their label-only attacks supports their practical relevance.

The empirical family — Nasr et al. (2018) adversarial regularization and Jia et al. (2019) MemGuard — targets the accuracy cost of DP-SGD by instead defending only against a learned attacker model. Adversarial regularization modifies training so that the classifier and an inference adversary are trained jointly; MemGuard operates purely on released confidences and converts them into adversarial examples against the attacker. Both achieve strong numbers within their own threat models, but both are precisely the kind of defense that Carlini et al.'s first-principles critique implicitly targets: strong average-case resistance need not imply resistance at the low-FPR regime, and heuristic defenses that depend on a specific attacker class are historically vulnerable to adaptive attacks. Reconciling this tension — how to obtain empirical defenses that also hold up under low-FPR evaluation — remains an open problem.

### 4.4 LLM-era extensions
Carlini et al. (2021) and Mattern et al. (2023) extend the MIA agenda into large language models, but with different emphases. Carlini et al. tackle an extraction problem: rather than deciding whether a particular sequence is in the training data of GPT-2, they generate candidate sequences and use membership-inference–style ranking to recover memorized text verbatim. The finding that even sequences appearing only once in training can be extracted, and that larger models are more vulnerable, ties MIA research directly to concrete privacy harm — and motivates defenses against memorization at scale. Mattern et al. return to the classical decision problem but do so in a setting where reference-based attacks are unrealistic, because training distributions of modern LLMs are not reproducible. Their neighbourhood-comparison attack removes the reference-corpus assumption by generating synthetic neighbours and comparing scores locally, and achieves performance competitive with reference-based attacks that assume perfect distributional knowledge. The LLM setting also highlights an open disagreement in the field about how to calibrate MIA evaluation when any single training example has vanishingly small influence.

### 4.5 Surveys and analyses
Hu et al. (2022) provide the most widely cited taxonomy of MIAs, covering both discriminative and generative targets and laying out a defense taxonomy that matches the attack-side structure. For a thesis, its main value is as a consolidated reading list and as a source of a common vocabulary; its main limitation is its publication date, which places it just before the first-principles re-evaluation of metrics and the current wave of LLM-targeted attacks. Any review that relied on Hu et al. alone would therefore miss both Carlini et al. (2022) and Mattern et al. (2023).

## 5. Open Problems for a Master's Thesis

1. **Low-FPR evaluation of empirical defenses.** Carlini et al. (2022) argued that MIAs must be measured at very low false-positive rates to reflect real privacy harm, but earlier empirical defenses including adversarial regularization (Nasr et al., 2018) and MemGuard (Jia et al., 2019) were primarily validated on aggregate metrics. Systematically re-evaluating these defenses under a LiRA-style low-FPR protocol — and testing them against adaptive attackers aware of the defense — is a well-scoped thesis contribution.

2. **Closing the DP-SGD accuracy gap beyond vision.** Papernot et al. (2021) recovered substantial accuracy for DP-SGD on small image benchmarks by replacing ReLUs with tempered sigmoids, but scaling that insight to large language models, transformer architectures, or tabular healthcare data remains open. A thesis could test whether similar architecture-level interventions narrow the DP accuracy gap in a non-vision setting.

3. **MIAs under realistic query budgets and rate limits.** Choquette-Choo et al. (2021) showed that label-only attacks can be as effective as confidence-based ones, but typically at the cost of many queries per record. No paper in the review analyzes how realistic API rate limits interact with attack success. A thesis could characterize the query-budget frontier at which current attacks stop being useful and explore whether server-side query budgeting is a viable lightweight defense.

4. **Reference-free MIA for LLMs with quantified confidence.** Mattern et al. (2023) showed that neighbourhood comparison removes the reference-corpus assumption, but the calibration of their scores at low false-positive rates on modern instruction-tuned models is still poorly characterized. Reconciling Mattern et al.'s reference-free framing with Carlini et al.'s (2022) first-principles metric is an open and timely problem.

5. **Defense against extraction vs. defense against decision MIAs.** Carlini et al. (2021) highlighted that memorization in LLMs produces verbatim extraction, which is a qualitatively different threat from the classical yes/no membership decision. It remains unclear whether DP-SGD (Abadi et al., 2016) at practical epsilons meaningfully reduces verbatim extraction, or only the aggregate MIA signal. A thesis could measure both under the same training recipe.

6. **Federated learning with active adversaries.** Nasr, Shokri, and Houmansadr (2019) demonstrated that active participants in federated learning can run membership inference against other participants' updates. Practical defenses — beyond vanilla DP — that cope with active adversaries who can influence the global model are still sparse in the literature covered here, and a thesis could benchmark secure aggregation, clipping, and DP variants against this stronger adversary.

## 6. Common Limitations Across the Field

- **Overreliance on small vision benchmarks.** Most of the attacks and defenses above are evaluated on CIFAR-10, MNIST, or similar, which limits how confidently the findings transfer to tabular healthcare data, audio, or large language models.
- **Aggregate metrics mask individual harm.** As Carlini et al. (2022) argue, reporting balanced accuracy or AUC can make defenses look stronger than they are when viewed through the lens of low-FPR attacks on specific records.
- **Empirical defenses vs. adaptive attackers.** Defenses that are evaluated only against a fixed attack class (for example, MemGuard against its known attack model) are historically fragile once the attacker adapts to the defense.
- **Unrealistic distributional assumptions.** Many attacks implicitly assume the adversary can sample from a distribution close to the target's training data; this is especially untenable for modern LLMs, as Mattern et al. (2023) point out.
- **Limited engagement with deployment constraints.** The literature rarely integrates API rate limits, query logging, or monitoring as part of the threat model, even though these are first-line defenses in production.
- **DP accuracy cost is still significant off-vision.** Even with innovations like tempered sigmoids, DP-SGD can meaningfully hurt accuracy on harder or non-vision datasets, leaving defenders with an uncomfortable trade-off.
- **Weak standardization of MIA evaluation.** Different papers use different shadow-model counts, different splits, and different metrics, making cross-paper comparison difficult — a recurring concern that Hu et al. (2022) also flag.

## 7. References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. In *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security* (pp. 308–318). ACM. https://arxiv.org/abs/1607.00133

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership inference attacks from first principles. In *2022 IEEE Symposium on Security and Privacy*. https://arxiv.org/abs/2112.03570

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., & Raffel, C. (2021). Extracting training data from large language models. In *30th USENIX Security Symposium*. https://arxiv.org/abs/2012.07805

Choquette-Choo, C. A., Tramer, F., Carlini, N., & Papernot, N. (2021). Label-only membership inference attacks. In *Proceedings of the 38th International Conference on Machine Learning*. https://arxiv.org/abs/2007.14321

Hu, H., Salcic, Z., Sun, L., Dobbie, G., Yu, P. S., & Zhang, X. (2022). Membership inference attacks on machine learning: A survey. *ACM Computing Surveys, 54*(11s), Article 235. https://arxiv.org/abs/2103.07853

Jia, J., Salem, A., Backes, M., Zhang, Y., & Gong, N. Z. (2019). MemGuard: Defending against black-box membership inference attacks via adversarial examples. In *Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security*. https://arxiv.org/abs/1909.10594

Long, Y., Bindschaedler, V., Wang, L., Bu, D., Wang, X., Tang, H., Gunter, C. A., & Chen, K. (2018). *Understanding membership inferences on well-generalized learning models*. arXiv. https://arxiv.org/abs/1802.04889

Mattern, J., Mireshghallah, F., Jin, Z., Schölkopf, B., Sachan, M., & Berg-Kirkpatrick, T. (2023). Membership inference attacks against language models via neighbourhood comparison. In *Findings of the Association for Computational Linguistics: ACL 2023*. https://arxiv.org/abs/2305.18462

Nasr, M., Shokri, R., & Houmansadr, A. (2018). Machine learning with membership privacy using adversarial regularization. In *Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security*. https://arxiv.org/abs/1807.05852

Nasr, M., Shokri, R., & Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In *2019 IEEE Symposium on Security and Privacy*. https://arxiv.org/abs/1812.00910

Papernot, N., Thakurta, A., Song, S., Chien, S., & Erlingsson, U. (2021). Tempered sigmoid activations for deep learning with differential privacy. In *Proceedings of the AAAI Conference on Artificial Intelligence*. https://arxiv.org/abs/2007.14191

Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M., & Backes, M. (2019). ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models. In *Network and Distributed System Security Symposium (NDSS)*. https://arxiv.org/abs/1806.01246

Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. In *2017 IEEE Symposium on Security and Privacy*. https://arxiv.org/abs/1610.05820

Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy risk in machine learning: Analyzing the connection to overfitting. In *2018 IEEE 31st Computer Security Foundations Symposium*. https://arxiv.org/abs/1709.01604

## 8. Disclosure

This literature review was prepared with assistance with Ai gammer check and enhance writing also for source retrieval; all citations were verified against publicly available abstracts before inclusion.
