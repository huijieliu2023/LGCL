## Codes for "From Intra- to Inter-domain: LLM-enhanced Graph Contrastive Learning for Technological Knowledge Flow Prediction"

## The architecture of LGC-TKF framework
![frame](https://github.com/user-attachments/assets/764dffb5-4c92-455a-ba24-a0ff46fd03cc)

## Dataset

Raw data:  https://patentsview.org/download/data-download-tables

Processed Data:


CPC-L3: data/CPC-L3


Due to upload capacity limitations, the CPC-R and CPC-G datasets will be made publicly available upon acceptance of the paper.


## Statistics of Datasets from 2010 to 2020, where $\beta$ represents homophily ratio.
| Dataset | \# Nodes | \# TKFs | \# TCs | \# Domains | $\beta$ |
|---------|----------|---------|--------|------------|--------|
| CPC-L3  | 678      | 60,153  | 32,451 | 9          | 0.26   |
| CPC-R   | 20,000   | 460,460 | 125,126| 9          | 0.58   |
| CPC-G   | 38,847   | 6,332,330| 2,313,910| 1        | 1      |


## Overall Performance Evaluation on Different Datasets (%). Notably, OOM indicates the out-of-memory issue.
<img width="885" alt="截屏2025-02-02 23 06 45" src="https://github.com/user-attachments/assets/e0cf08af-dc65-4b96-86f8-1655969a5288" />

