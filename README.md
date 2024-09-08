# Repository for Paper "Spatio-temporal Traffic Accidents Detection via Graph based Generative Adversarial Network"

This repository contains the code and data for the paper titled "Spatio-temporal Traffic Accidents Detection via Graph based Generative Adversarial Network".

## Contents

- **data/**: This folder contains the following:
  - Traffic flow data from Bay Area, dated [Dec, 2016].
  - Adjacency matrix of the traffic network in the region.

- **model/**: This folder contains the following files:
  - `module.py`: Defines the primary components and architecture of the model.
  - `gan.py`: Includes the main implementation of the model.
  - `utils.py`: Provides functions for data preprocessing.
  - `train.py`: Contains code for setting up the training process.

To train the model, please run the `train.py` file.

## Additional Data

For traffic accidents data, you can access it via the following Google Drive link: [https://drive.google.com/drive/folders/1SLkpl2MI4saRX7h02qMcDrnU2soM-d_b?usp=sharing](https://drive.google.com/drive/folders/1SLkpl2MI4saRX7h02qMcDrnU2soM-d_b?usp=sharing).

For more information and additional data, please visit [https://pems.dot.ca.gov/](https://pems.dot.ca.gov/).


## Citation

If you are interested in our research, please cite it as follows:

```bibtex
@misc{acc_detection,
  author       = {Lyuyi ZHU, Qixin ZHANG, Xiangru Jian, Lishuai LI, Yu YANG},
  title        = {Spatio-temporal Traffic Accidents Detection via Graph based Generative Adversarial Network},
  year         = {2024}
}
