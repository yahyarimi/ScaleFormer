# SCALE-Former Recent Reference Audit (2022–2026)

Purpose: establish a recent, defensible literature pool for the SCALE-Former manuscript before generating the final BibTeX file. This audit intentionally separates closely related CSI-feedback/channel-estimation work from the paper's own sparse measured-CSI reconstruction problem.

## Selection rules

- Main recent window: 2022–2026.
- Priority: peer-reviewed IEEE/Elsevier/Springer/Nature/Wiley/IET publications and established conference proceedings.
- Include works on CSI feedback/reconstruction, channel estimation, Transformers/attention, scalable/adaptive CSI learning, efficiency/quantization, and measured/real-world validation.
- Foundational pre-2022 papers will be kept separately and will not be counted toward the recent-reference target.
- Inclusion in this audit does not mean every paper must appear in the final manuscript; references should ultimately be cited only where they support a specific statement.

## Recent reference pool

| # | Year | Reference | Venue | DOI | Role in SCALE-Former |
|---:|---:|---|---|---|---|
| 1 | 2022 | Guo et al., **Overview of Deep Learning-Based CSI Feedback in Massive MIMO Systems** | IEEE Transactions on Communications | 10.1109/TCOMM.2022.3217777 | Recent survey / CSI-feedback landscape |
| 2 | 2022 | Chen et al., **Deep Learning-Based Implicit CSI Feedback in Massive MIMO** | IEEE Transactions on Communications | 10.1109/TCOMM.2021.3138097 | Practical implicit feedback framing |
| 3 | 2022 | Guo et al., **CSI Feedback With Model-Driven Deep Learning of Massive MIMO Systems** | IEEE Communications Letters | 10.1109/LCOMM.2021.3138927 | Model-driven feedback / unfolding |
| 4 | 2022 | Tang et al., **Dilated Convolution Based CSI Feedback Compression for Massive MIMO Systems** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2022.3183596 | CNN reconstruction baseline context |
| 5 | 2022 | Sun et al., **Deep Learning-Based Joint CSI Feedback and Hybrid Precoding in FDD mmWave Massive MIMO Systems** | Entropy | 10.3390/e24040441 | End-to-end feedback / precoding context |
| 6 | 2022 | Zhang et al., **Fully Connected Layer-Shared Network Architecture for Massive MIMO CSI Feedback** | Electronics Letters | 10.1049/ell2.12418 | Cross-scenario parameter sharing |
| 7 | 2022 | Shen et al., **MixerNet: Deep Learning for Eigenvector-Based CSI Feedback** | WCSP 2022 | 10.1109/WCSP55476.2022.10039349 | Variable/eigenvector feedback context |
| 8 | 2022 | Cui et al., **TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System** | IEEE Wireless Communications Letters | 10.1109/LWC.2022.3149416 | Full-attention Transformer comparator |
| 9 | 2022 | Peng et al., **Deep Learning Based Channel Estimation for OFDM Systems With Doubly Selective Channel** | IEEE Communications Letters | 10.1109/LCOMM.2022.3187161 | Recent OFDM channel-estimation context |
| 10 | 2023 | Mourya et al., **A Spatially Separable Attention Mechanism for Massive MIMO CSI Feedback** | IEEE Wireless Communications Letters | 10.1109/LWC.2022.3216352 | Efficient Transformer attention |
| 11 | 2023 | Guo et al., **Multi-Task Training Approach for CSI Feedback in Massive MIMO Systems** | IEEE Communications Letters | 10.1109/LCOMM.2022.3217059 | Multi-scenario training |
| 12 | 2023 | **Deep Learning-Based Massive MIMO Channel Estimation With Reduced Feedback** | Digital Signal Processing | 10.1016/j.dsp.2023.104009 | Joint estimation / reduced feedback |
| 13 | 2023 | Kim et al., **Transformer-Based Channel Parameter Acquisition for Terahertz Ultra-Massive MIMO Systems** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2023.3287530 | Transformer channel acquisition |
| 14 | 2023 | **A Deep Learning Method for Joint Compression and Unsupervised Denoising of CSI Feedback** | IEEE ICC 2023 | 10.1109/ICC45041.2023.10279775 | Noisy CSI / reconstruction loss context |
| 15 | 2023 | Shan et al., **Attention Guided Deep Learning for CSI Compression and Feedback** | IEEE ICCT 2023 | 10.1109/ICCT59356.2023.10419429 | Attention-based CSI reconstruction |
| 16 | 2023 | Guo et al., **Deep Learning-Based Two-Timescale CSI Feedback for Beamforming Design in RIS-Assisted Communications** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2022.3226220 | Structured/two-timescale CSI feedback |
| 17 | 2024 | **Low-Complexity Deep Learning Framework for CSI Feedback in Massive MIMO System** | IEEE Wireless Communications Letters | 10.1109/LWC.2023.3347573 | Low-complexity CNN feedback baseline |
| 18 | 2024 | Guo et al., **Deep Learning for Joint Channel Estimation and Feedback in Massive MIMO Systems** | Digital Communications and Networks | 10.1016/j.dcan.2023.01.011 | Joint estimation/reconstruction framing |
| 19 | 2024 | Hu et al., **Multiple-Compression-Rate CSI Feedback for mm-Wave Massive MIMO Based on Federated Learning** | Physical Communication | 10.1016/j.phycom.2024.102305 | Multiple-rate / federated feedback |
| 20 | 2024 | Zhang et al., **Continuous Online Learning-Based CSI Feedback in Massive MIMO Systems** | IEEE Communications Letters | 10.1109/LCOMM.2024.3350210 | Distribution shift / continual adaptation |
| 21 | 2024 | Li et al., **Facilitating AI-Based CSI Feedback Deployment in Massive MIMO Systems With Learngene** | IEEE Transactions on Wireless Communications | 10.1109/TWC.2024.3381211 | Deployment / transfer learning |
| 22 | 2024 | Zhou et al., **A Low-Overhead Incorporation-Extrapolation Based Few-Shot CSI Feedback Framework for Massive MIMO Systems** | IEEE Transactions on Wireless Communications | 10.1109/TWC.2024.3418612 | Few-shot feedback / extrapolation |
| 23 | 2024 | Guo et al., **Learning-Based Integrated CSI Feedback and Localization in Massive MIMO** | IEEE Transactions on Wireless Communications | 10.1109/TWC.2024.3422399 | Multi-task CSI representation |
| 24 | 2024 | Xu et al., **Disentangled Representation Learning Empowered CSI Feedback Using Implicit Channel Reciprocity in FDD Massive MIMO** | IEEE Transactions on Wireless Communications | 10.1109/TWC.2024.3426525 | Representation learning / reciprocity |
| 25 | 2024 | Cui et al., **Lightweight Neural Network With Knowledge Distillation for CSI Feedback** | IEEE Transactions on Communications | 10.1109/TCOMM.2024.3377724 | Accuracy-efficiency trade-off |
| 26 | 2024 | Shin et al., **Vector Quantization for Deep-Learning-Based CSI Feedback in Massive MIMO Systems** | IEEE Wireless Communications Letters | 10.1109/LWC.2024.3415633 | Finite-bit feedback / VQ |
| 27 | 2024 | Shao et al., **Quantized Trainable Compressed Sensing for MIMO CSI Feedback** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2024.3446464 | Quantized low-complexity reconstruction |
| 28 | 2024 | **Multi-Head Transformer Architecture with Higher Dimensional Feature Representation for Massive MIMO CSI Feedback (TransNet+)** | Applied Sciences | 10.3390/app14041356 | Transformer CSI reconstruction |
| 29 | 2024 | Xiao et al., **AI Enlightens Wireless Communication: A Transformer Backbone for CSI Feedback** | China Communications | 10.23919/JCC.ea.2022-0186.202401 | Transformer/eigenvector CSI feedback |
| 30 | 2024 | Yue et al., **Transformer-Empowered Receiver Design of OFDM Communication Systems** | Computer Communications | 10.1016/j.comcom.2024.107960 | Transformer receiver / OFDM context |
| 31 | 2024 | Tamiru et al., **Channel Estimation and Pilot Reduction for mmWave Massive MIMO Systems Using Deep Neural Networks** | ICT Express | 10.1016/j.icte.2024.02.003 | Reduced-pilot channel estimation |
| 32 | 2024 | Kim et al., **Deep Learning-Aided Parametric Sparse Channel Estimation for Terahertz Massive MIMO Systems** | IEEE Transactions on Cognitive Communications and Networking | 10.1109/TCCN.2024.3401710 | Sparse parametric estimation |
| 33 | 2024 | Inoue et al., **DTL-Based CSI Feedback Combined With Continual Learning in FDD Massive MIMO Systems** | ICNC 2024 | 10.1109/ICNC59896.2024.10556096 | Adaptation / catastrophic forgetting |
| 34 | 2024 | Jameel et al., **Deep OFDM Channel Estimation: Capturing Frequency Recurrence** | IEEE Communications Letters | 10.1109/LCOMM.2024.3350369 | Frequency-structure modeling |
| 35 | 2024 | Shen et al., **An Adaptive CSI Feedback Model Based on BiLSTM for Massive MIMO-OFDM Systems** | IEEE Internet of Things Journal | 10.1109/JIOT.2024.3436024 | Variable input length / variable feedback |
| 36 | 2025 | Yin et al., **Joint Channel Estimation and Feedback With Masked Token Transformers in Massive MIMO Systems** | Scientific Reports | 10.1038/s41598-025-25256-1 | Masked-token Transformer / completion |
| 37 | 2025 | **Transformer-Based Site-Specific Channel Estimation** | IEEE WCNC 2025 | 10.1109/WCNC61545.2025.10978364 | Transformer channel inference |
| 38 | 2025 | Shehzad et al., **Design of an Efficient CSI Feedback Mechanism in Massive MIMO Systems: A Machine Learning Approach Using Empirical Data** | IEEE Transactions on Cognitive Communications and Networking | 10.1109/TCCN.2024.3435915 | Empirical/measured CSI validation |
| 39 | 2025 | Huang et al., **AI-Based CSI Feedback With Digital Twins: Real-World Validation and Insights** | IEEE Wireless Communications Letters | 10.1109/LWC.2025.3570738 | Simulation-to-real / real-world validation |
| 40 | 2025 | **Fusion of Self-Attention Mechanism for CSI Feedback in Massive MIMO Systems (FSAMNet)** | ICT Express | 10.1016/j.icte.2024.10.010 | Self-attention CSI feedback |
| 41 | 2025 | Xiang et al., **JCANet: Multi-Domain Federated Lightweight Self-Attention CSI Feedback Network** | Physical Communication | 10.1016/j.phycom.2025.102682 | Lightweight self-attention / federated learning |
| 42 | 2025 | Qing et al., **Perception-Assisted Superimposed CSI Feedback for Millimeter Wave Massive MIMO Systems** | Physical Communication | 10.1016/j.phycom.2025.102864 | Sensing/perception-assisted feedback |
| 43 | 2025 | **A Recursive Discretization Compression Framework Combined With Selective State Space Model for Massive MIMO CSI Feedback** | IEEE VTC 2025-Spring | 10.1109/VTC2025-Spring65109.2025.11174530 | State-space / Mamba-style CSI modeling |
| 44 | 2025 | Cen et al., **When Transformer Meets CSI Feedback in mMIMO Systems: A Lightweight CsiMobileViT Approach** | IEEE VTC 2025-Fall | 10.1109/VTC2025-Fall65116.2025.11310376 | Lightweight Transformer feedback |
| 45 | 2025 | **LECA-Net: A Context-Aware Local Enhancement Network for Massive MIMO CSI Feedback** | IEEE INFOCOM Workshops 2025 | 10.1109/INFOCOMWKSHPS65812.2025.11152839 | Local feature enhancement |
| 46 | 2025 | Shin et al., **Entropy-Constrained VQ-VAE for Deep-Learning-Based CSI Feedback** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2025.3542267 | Entropy-constrained finite-bit feedback |
| 47 | 2025 | **Deep Joint CSI Feedback and Multiuser Precoding for MIMO OFDM Systems** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2024.3452409 | End-to-end CSI feedback / precoding |
| 48 | 2025 | Cui et al., **Efficient Deep Learning-Based Cascaded Channel Feedback in RIS-Assisted Communications** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2024.3461830 | Structured channel compression |
| 49 | 2025 | **Performance-Guaranteed CSI Feedback via Model-Agnostic Incremental Residual Compression Framework** | IEEE Wireless Communications Letters | 10.1109/LWC.2025.3624978 | Residual compression / worst-case control |
| 50 | 2025 | Zhang and Alkhateeb, **Zone-Specific CSI Feedback for Massive MIMO: A Situation-Aware Deep Learning Approach** | IEEE Wireless Communications Letters | 10.1109/LWC.2024.3457896 | Spatial distribution shift / deployment |
| 51 | 2025 | Zhao et al., **Enhanced Meta-Transfer Learning Assisted CSI Feedback in Massive MIMO Systems** | IEEE Wireless Communications Letters | 10.1109/LWC.2024.3511553 | Scenario adaptation / meta-transfer |
| 52 | 2025 | Pal et al., **Performance Evaluation of AI-Based CSI Feedback Schemes Compliant With 3GPP Standards** | Physical Communication | 10.1016/j.phycom.2025.102803 | 3GPP-aligned evaluation |
| 53 | 2025 | Li et al., **CSI Feedback for Massive MIMO System With Joint Attention Mechanism and Multi-Scale Convolution** | International Journal of Computational Intelligence Systems | 10.1007/s44196-025-01008-x | Attention + local multiscale modeling |
| 54 | 2025 | **Channel Estimation in Massive MIMO-OFDM With Multi-Group Adjustable Phase Shift Pilots** | IEEE VTC 2025-Spring | 10.1109/VTC2025-Spring65109.2025.11174505 | Sparse-pilot channel estimation |
| 55 | 2025 | Ibwe, **Sparse Hybrid-Field Channel Estimation in Massive MIMO Systems** | Journal of Electrical Systems and Information Technology | 10.1186/s43067-025-00277-y | Sparse channel estimation context |
| 56 | 2025 | Ma et al., **A Review of Channel Estimation Research Methods for Massive MIMO Systems** | Physical Communication | 10.1016/j.phycom.2025.102632 | Recent channel-estimation review |
| 57 | 2025 | Gao et al., **Deep Residual Learning With Attention Mechanism for OFDM Channel Estimation** | IEEE Wireless Communications Letters | 10.1109/LWC.2022.3232378 | Attention + residual channel estimation |
| 58 | 2025 | Kumar et al., **A Novel DRT-Assisted Deep Learning-Based Channel Feedback Mechanism for Massive MIMO** | IEEE Transactions on Consumer Electronics | 10.1109/TCE.2024.3520220 | Transform-domain CSI feedback |
| 59 | 2026 | **Large Language Model Empowered CSI Feedback in Massive MIMO Systems** | IEEE Transactions on Wireless Communications | 10.1109/TWC.2026.3671802 | Latest foundation/LLM-based CSI feedback |
| 60 | 2026 | Kim et al., **Combining Spectral and Inception Transformer for Massive MIMO Channel State Information Feedback (SPINNet)** | ETRI Journal | 10.4218/etrij.2025-0427 | Local-global efficient Transformer design |
| 61 | 2026 | Yoon et al., **CsiMamba: Efficient Channel State Information Feedback for FDD Massive MIMO Systems** | ETRI Journal | 10.4218/etrij.2025-0542 | State-space global/local sequence modeling |
| 62 | 2026 | Pan et al., **GMSANet: A Hybrid CNN-Transformer Network for CSI Feedback** | Physical Communication | 10.1016/j.phycom.2026.103010 | Hybrid CNN-Transformer feedback |
| 63 | 2026 | Liu et al., **Generative Model-Aided Continual Learning for CSI Feedback in FDD mMIMO-OFDM Systems** | IEEE Signal Processing Letters | 10.1109/LSP.2026.3665655 | Continual learning / scenario adaptation |
| 64 | 2026 | Shen et al., **DL-Based Adaptive Joint CSI Prediction and Feedback for Massive MIMO-OFDM Systems** | IEEE Transactions on Vehicular Technology | 10.1109/TVT.2026.3670823 | Variable dimension / temporal adaptation |
| 65 | 2026 | **Large Transformer Model With Haar Wavelet Transform for Channel Estimation in Massive MIMO Systems** | Internet Technology Letters | 10.1002/itl2.70204 | Transformer channel estimation |
| 66 | 2026 | Fairuz et al., **Hybrid Transformer-Based Channel Estimation for Massive MIMO Systems Using Low-Rank Adaptation Optimization** | IntechOpen research article | 10.5772/acrt20250087 | Transformer adaptation / LoRA |
| 67 | 2026 | **Channel Estimation for MIMO-OFDM Based on Adaptive Transformer Network in High-Speed Mobile Scenarios (AdaFiT)** | Journal of Electronics & Information Technology | 10.11999/JEIT260075 | Adaptive Transformer channel estimation |

## Count

- 2022: 9
- 2023: 7
- 2024: 19
- 2025: 23
- 2026: 9
- **Recent 2022–2026 pool: 67 references**

This exceeds the manuscript requirement of at least 40 recent references. The final paper should normally cite a smaller, purpose-driven subset (roughly 45–55 recent papers) plus a handful of foundational references rather than citing all 67 indiscriminately.

## Foundational references to retain separately (pre-2022)

1. Vaswani et al., **Attention Is All You Need**, NeurIPS 2017.
2. Ye, Li, and Juang, **Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems**, IEEE Wireless Communications Letters, 2018, DOI 10.1109/LWC.2017.2757490.
3. Wen, Shih, and Jin, **Deep Learning for Massive MIMO CSI Feedback (CsiNet)**, IEEE Wireless Communications Letters, 2018, DOI 10.1109/LWC.2018.2818160.
4. Yang, Boloursaz Mashhadi, and Gündüz, **Deep Convolutional Compression for Massive MIMO CSI Feedback (DeepCMC)**, IEEE MLSP 2019, DOI 10.1109/MLSP.2019.8918798.
5. Boloursaz Mashhadi and Gündüz, **Deep Learning for Massive MIMO Channel State Acquisition and Feedback**, Journal of the Indian Institute of Science, 2020, DOI 10.1007/s41745-020-00169-2.
6. Euchner et al., **A Distributed Massive MIMO Channel Sounder for “Big CSI Data”-Driven Machine Learning**, WSA 2021.
7. Euchner and Gauger, **CSI Dataset dichasus-005x: Indoor Non-Line of Sight, Single Room**, DaRUS, 2021, DOI 10.18419/DARUS-2204.

## SCALE-Former citation guardrails

- Do not use the existence of many Transformer papers to claim that SCALE-Former is the first Transformer for CSI estimation or feedback.
- Distinguish CSI feedback/compression from sparse-observation reconstruction throughout the Introduction and Related Work.
- Use DICHASUS as measured reference CSI; do not call it a physical ground-truth channel.
- Do not describe controlled sparse observation masks as operational DICHASUS pilot patterns.
- Use the scalability literature to motivate flexibility, but claim only transfer across the tested B/F configurations without retraining.
- Preserve the measured accuracy-efficiency trade-off: SCALE-Former is more accurate than the parameter-comparable frequency Transformer but uses more MACs, memory, and latency.
- Retain the block-mask failure and the no-global capacity confound in the manuscript.
