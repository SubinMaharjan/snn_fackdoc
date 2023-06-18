
*Repository for the project of identity documents authentication (id-FDGP-1)*

Any use of this repository (codes, dataset) is required to cite the following reference:

M. Al-Ghadi, Z. Ming, P. Gomez-Krämer, and J.C. Burie. Identity documents authentication based on forgery detection of guilloche pattern, arXiv, 2022.



*Introduction*

Identity documents are always including more and more sophisticated security features in their designs in order to ward off potential counterfeiters, fraudsters and impostors. One of these security features is the Guilloche. The Guilloche design is a pattern of computer-generated fine lines that forms a unique shape. The target is to develop detection and verification approach of the Guilloche pattern in order to ensure the authenticity of the identity documents.

Description of files in this repository


*Codes (.py)*:

- training_id-FDGP-1.py: trains the Siames network() for the training data set of MIDV-2020 countrie. The output of this script is a trained model (.ckpt) for each country of MIDV-2020.

- loss.py: defines the Contrastive loss function.

- data_pairs.py: have SiameseNetworkDataset() class which parining between input, and have SiameseNetwork() class which extracts the feature vector of each input pair.

- testing_id-FDGP-1.py: gets the distances distribution for the testing data set of MIDV-2020 countries and their ROC curves.


*models (.ckpt)*:
- contains the trained models for 10 countries of MIDV-2020; we introduce only one model (which is trained model of alb country) due to space constraints.


*FMIDV dataset*:
- A dataset contains 28k forged IDs for 10 countries based on copy-move forgeries on the identity documents of  MIDV-2020 dataset.

- The dataset has a size of 4,7 GB and is hosted on an FTP server of the University of La Rochelle. Please fill this form (https://forms.office.com/r/gVsSivTFYz) for getting access to the dataset. In  case of any problem, please contact (musab.alghadi@univ-lr.fr or  muhammad_muzzamil.luqman@univ-lr.fr).

