# MEGPE: Multi-Expert Genetic Programming based Ensemble for Long-Tailed Image Classification

This repository implements MEGPE as presented in the IEEE TEVC 2026 paper. MEGPE is a novel genetic programming method for long-tailed image classification task.

<img width="2754" height="1080" alt="image" src="https://github.com/user-attachments/assets/0dd76449-e90c-498c-a88f-39732d51fe92" />

# Abstract
Long-tailed image classification faces challenges of data imbalance and poor feature representation for tail classes, leading to biased predictions favoring head classes. While most existing methods rely on deep neural networks (DNNs), they typically require large amounts of training data and lack interpretability. Genetic Programming (GP) has shown promise in few-shot learning but has seldom been investigated in long-tailed image classification, primarily due to its limited ability to handle class imbalance and its tendency for fitness functions to be biased toward head classes. To fill this gap, this paper proposes a multi-expert GP method for long-tailed image classification. We develop three objective functions, each serving as an expert: 1) a long-tailed expert focusing on head-class performance; 2) a balanced-class expert that promotes equal class representation; and 3) an inverse long-tailed expert emphasizing tail classes. This tri-expert framework enables GP to jointly optimize complementary objectives and learn robust feature representations for both head and tail classes. To further improve classification performance, the evolved GP individuals from the final population are used to train base learners, and their outputs are integrated via a voting-based ensemble model. Experimental results demonstrate that the proposed method outperforms state-of-the-art GP and DNN approaches without pretraining across seven long-tailed image classification datasets.

# Acknowledge
Please kindly cite this paper in your publications if it helps your research:
```
@article{chen2026multi,
title={Multi-Expert Genetic Programming based Ensemble for Long-Tailed Image Classification},
author={Chen, Zhuoya and Fan, Qinglan and Jiao, Ruwang and Xue, Bing and Hunag, He and Dai, Yifan and Zhang, Mengjie},
journal={IEEE Transactions on Evolutionary Computation},
year={2026},
publisher={IEEE}
}
```

# License
This code is released under the MIT License.

# Contact
If you face any difficulty with the implementation, please refer to: zhuoyachen00@gmail.com
