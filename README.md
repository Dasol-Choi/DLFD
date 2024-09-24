## Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2310.04313-b31b1b.svg)](https://arxiv.org/abs/2310.04313) -->

### Authors
[Dasol Choi](https://github.com/Dasol-Choi), [Donbin Na](https://github.com/ndb796)

### Abstract
> With the explosive growth of deep learning applications, the right to be forgotten has become increasingly in demand in various AI industries. For example, given a facial recognition system, some individuals may wish to remove images that might have been used in the training phase from the trained model. Unfortunately, modern deep neural networks sometimes unexpectedly leak personal identities. Recent studies have presented various machine unlearning algorithms to make a trained model unlearn the data to be forgotten. While these methods generally perform well in terms of forgetting scores, we have found that an unexpected model utility drop can occur. This phenomenon, which we term correlation collapse, happens when the machine unlearning algorithms reduce the useful correlation between image features and the true label. To address this challenge, we propose Distribution-Level Feature Distancing (DLFD), a novel method that efficiently forgets instances while preventing correlation collapse. Our method synthesizes data samples so that the generated data distribution is far from the distribution of samples being forgotten in the feature space, achieving effective results within a single training epoch. Through extensive experiments on facial recognition datasets, we demonstrate that our approach significantly outperforms state-of-the-art machine unlearning methods.

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
