\# SCALE-Former



SCALE-Former is a scalable transformer framework for sparse channel reconstruction from real measured massive-MIMO channel state information (CSI).



\## Research Objective



This project investigates reconstruction of dense measured CSI from sparse antenna-frequency observations using real wireless channel measurements.



The primary dataset is the DICHASUS-005x measurement campaign.



\## Channel Representation



Each reference channel measurement is represented as a complex antenna-frequency matrix:



H ∈ C^(B × F)



where:



\- B is the number of receive antennas.

\- F is the number of frequency-domain channel coefficients.



Sparse observations are generated using an observation mask:



H\_obs = M ⊙ H



The estimator predicts:



H\_hat = f(H\_obs, M)



The objective is to reconstruct the complete measured channel from sparse observations.



\## Primary Dataset



DICHASUS-005x



Planned data split:



\- Training: measurement runs 0053, 0054, 0055

\- Validation: measurement run 0056

\- Test: measurement run 0057



The split is performed by independent measurement runs rather than randomly mixing neighboring CSI samples.



\## Planned Experiments



\- Real measured CSI reconstruction

\- Observation-density robustness

\- Observation-pattern generalization

\- Frequency-size scalability

\- Antenna-count scalability

\- Cross-run generalization

\- Multi-seed stability analysis

\- Parameter-count analysis

\- GFLOPs analysis

\- GPU-memory analysis

\- Inference-latency analysis



\## Environment



\- Operating system: Windows

\- Python: 3.13.9

\- PyTorch: 2.10.0+cu130

\- CUDA runtime: 13.0

\- GPU: NVIDIA GeForce RTX 3050 Laptop GPU

\- GPU memory: 4 GB



\## Data Storage



Raw datasets are not included in this repository.



DICHASUS data should be stored locally under:



data/dichasus\_005x/



\## Reproducibility



Training configurations, random seeds, evaluation metrics, experiment logs, result CSV files, and plotting scripts will be version controlled.



Raw datasets and large model checkpoints will not be committed to GitHub.

