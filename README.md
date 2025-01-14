# Fork of A PyTorch implementation of Molecular VAE paper Functional as of Jan 2025 but model performs very poorly. 
## Fixes listed at end of readme 



PyTorch implementation of the paper **"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" by Gómez-Bombarelli, et al.**\
Link to Paper - [arXiv](https://arxiv.org/abs/1610.02415)
<br />

<div style="text-align:center"><img src="https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch/blob/master/Sample_imgs/cover_img.jpg" /></div>

----

## Getting the Repo
To clone the repo on your machine run -\
`git clone https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch.git`\
The Structure of the Repo is as follows -\
`data_prep.py`- For Getting the Data in CSV format and splitting into specifed sized Train and Val\
`main.py` - Running the model\
`model.py` - Defines the Architecture of the Model\
`utils.py` - Various useful functions for encoding and decoding the data <br />


## Getting the Dataset
For this work I have used the ChEMBL Dataset which can be found [here](https://www.ebi.ac.uk/chembl/)\
\
Since the whole dataset has over 16M datapoints, I have decided to use a subset of that data.
To get the subset you can either use the train, val data present in ``/data``
or run the ``data_prep.py`` file as - \
`python data_prep.py /path/to/downloaded_data col_name_smiles /save/path 50000` \
\
This will prepare 2 CSV files `/save/path_train.csv` and `/save/path_val.csv` both of length 50k and having randomly shuffled datapoints.

Example of a Smiles string and corresponding Molecule



## Training the Network
To train the network use the `main.py` file

To Run the Papers Model (Conv Encoder and GRU Decoder)\
`python main.py ./data/chembl_500k_train ./data/chembl_500k_val ./Save_Models/ --epochs 100 --model_type mol_vae --latent_dim 290 --batch_size 512 --lr 0.0001`\
Latent Dim has default value 292 which is the value used in the original Paper

To Run a VAE with Fully Connected layers in both Encoder Decoder\
``python main.py ./data/bbbp.csv ./Save_Models/ --epochs 1 --model_type fc --latent_dim 100 --batch_size 20 --lr 0.0001``


## Results

The Train and Validation Losses where tracked for Training and Validation epochs

**Using Latent Dim = 292 (As in the Paper)** \
![Loss graphs](/Sample_imgs/graph_loss_1.png) 

It starts to overfit the train set after 20 Epochs, so the saved weights at 20 should be used for best results <br />



Although the Training Loss Reduces more in the 392 Case the Validation Loss remains almost equal which means it starts to overfit after 292.

### Sample Outputs

*Input* - \CC(C)(C)C(=O)OCN1OC(=O)c2ccccc12 \
*Output* - \CC(C)CC)C(=O)OC11CC(=O)C2ccccc12

*Input* - \CN\C(=N\S(=O)(=O)c1cc(CCNC(=O)c2cc(Cl)ccc2OC)ccc1OCCOC)\S \
*Output* - \CN\C(=N/S)=O)(=O)c1ccccCNC(=O)c2cc(Cl)ccc2OC)ccc1OCC(C(\C 

*Input* - \O[C@@H]1[C@@H](O)[C@@H](Cc2ccccc2)N(CCCCCNC(=O)c3ccccc3)C(=O)N(CCCCCNC(=O)c4ccccc4)[C@@H]1Cc5ccccc5 \
Output -  \O[C@@H]1[C@@H](O)[C@@H](Cc2ccccc2)N(CcCCCN3(=O)c3ccccc3)C(=O)N4Cc44NC4C=O)c4cccc54)c1Cc5ccccc5

*Input* - \C\C(=N/OC(c1ccccc1)c2ccccc2)\C[C@H]3CCc4c(C3)cccc4OCC(=O)O \
*Output* - \C\C(=N/OC(c1ccccc1)\2ccccc2)\C33CNC4ccc))ccc44OCC=O)O

*Input* - \O[C@@H](CNCCc1ccc(NS(=O)(=O)c2ccc(cc2)c3coc(n3)c4ccc(cc4)C(F)(F)F)cc1)c5cccnc5 \
*Output*- \O[C@@H](CNCCc1ccc(NS(=O)(=O)c2ccc(cc2)c3ncc(C3)C4cccccc4)C(F)(F)F)cc1)c5cccnc5 

*Input*- \CCCCCCCCCCc1cccc(O)c1C(=O)O \
*Output*- \CCCCCCCCCCc1ccccccc)CC(O))O 



# Fixes

Alterations to get VAE code working: 


fix some spaces to tabs

adjust python command to:
python main.py ./data/chembl_500k_train.csv ./data/chembl_500k_val.csv ./Save_Models/ --epochs 100 --model_type mol_vae --latent_dim 290 --batch_size 512 --lr 0.0001

latent_dim to args.latent_dim in main.py

from torch.nn import init in model.py

num_workers=0, in dataloader in main

Add self. to z = self._sample_latent(self.h_enc) model.py line 112ish

Add self.vocab_len = vocab_len to Conv_encoder in model.py

In GRU_Decoder 
in __init__:
add num_layers param like so:
def __init__(self, vocab_size,latent_dim, num_layers=3):
add self.num_layers = num_layers inside __init__
Also add self.embed = nn.Embedding(vocab_size, latent_dim)
change 501 to latent_dim in these two lines
self.gru = nn.GRU(latent_dim, 501, 3, batch_first=True)
self.fc_2 = nn.Linear(latent_dim, vocab_size)

in forward: change x_recon = self.fc_1(z_out) to x_recon = self.fc_2(z_out)

add import random to model.py

In Molecule_VAE 
in __init__ 
add parameter for teacher_forcing_ratio 
def __init__(self,encoder,decoder,device,latent_dim, teacher_forcing_ratio=0.5):
add definition too		
self.teacher_forcing_ratio = teacher_forcing_ratio

in main.py 
in two places:
change loss = F.binary_cross_entropy(input_recon, inputs, size_average=False) + latent_loss_val
to:
inputs_matched = match_shapes(input_recon, inputs)
loss = F.binary_cross_entropy(torch.sigmoid(input_recon), inputs_matched, reduction='sum') + latent_loss_val

and add matching helper function above them around line 155 just under "for i, data in enumerate(dataloader):"

In main.py
change 
if epoch_loss_recon_val < best_epoch_loss_val or epoch > 100: 
to 
if epoch_loss_val < best_epoch_loss_val or epoch > 100:

Reduced data size for faster debugging **TODO** put back later

Change  line 107ish in main.py to:
input_dim = 120 * vocab_size

in model.py in Encoder
in forward
add x = x.view(x.size(0), -1) like so: 
	def forward(self, x):
		x = x.view(x.size(0), -1) 
		z = F.relu(self.linear2(F.relu(self.linear1(x))))
		return z

in main.py
import os

after checkpoint definition (line 208ish)
add 
        if not os.path.exists(args.save_loc):
            os.makedirs(args.save_loc)


