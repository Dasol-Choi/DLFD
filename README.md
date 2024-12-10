## Distribution-Level Feature Distancing for Machine Unlearning
[![arXiv](https://img.shields.io/badge/arXiv-2409.14747-b31b1b.svg)](https://arxiv.org/abs/2409.14747) 
- This repository is the PyTorch implementation of our AAAI'2025 paper - [Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting](https://arxiv.org/abs/2409.14747)

### Authors
[Dasol Choi](https://github.com/Dasol-Choi), [Donbin Na](https://github.com/ndb796)

### Abstract
> With the explosive growth of deep learning applications and increasing privacy concerns, the right to be forgotten has become a critical requirement in various AI industries. For example, given a facial recognition system, some individuals may wish to remove their personal data that might have been used in the training phase. Unfortunately, deep neural networks sometimes unexpectedly leak personal identities, making this removal challenging. While recent machine unlearning algorithms aim to enable models to forget such data, we observe an unintended utility drop, termed correlation collapse, where these algorithms inadvertently weaken the essential correlations between image features and true labels during the forgetting process. To address this challenge, we propose Distribution-Level Feature Distancing (DLFD), a novel method that efficiently forgets instances while preserving task-relevant feature correlations. Our method synthesizes data samples by optimizing the feature distribution to be distinctly different from that of forget samples, achieving effective results within a single training epoch. Through extensive experiments on facial recognition datasets, we demonstrate that our approach significantly outperforms state-of-the-art machine unlearning methods in both forgetting performance and model utility preservation.

### Requirements

```bash
pip install -r requirements.txt
```

### Checkpoints

The `checkpoints/` directory contains pretrained models for the DLFD method and baseline models. You can load these checkpoints for further fine-tuning or evaluation:

- **DLFD Checkpoints:**
  - `age_dlfd.pth`: Checkpoint for the age classification task using the DLFD method.
  - `emotion_dlfd.pth`: Checkpoint for the emotion classification task using the DLFD method.

- **Original and Retrained Models:**
  - `last_100_age_original.pth`: Original model for age classification.
  - `last_100_age_retrained.pth`: Retrained model for age classification.
  - `last_100_emotion_original.pth`: Original model for emotion classification.
  - `last_100_emotion_retrained.pth`: Retrained model for emotion classification.

### Citation
<pre>
@misc{choi2024distributionlevelfeaturedistancingmachine,
      title={Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting}, 
      author={Dasol Choi and Dongbin Na},
      year={2024},
      eprint={2409.14747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14747}, 
}
