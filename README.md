# CRIPRR-RNA-Editing-Modelling

Run `pip install -r requirements.txt` to setup the env.

`CRISPR-RNA-Editing Model Training and Evaluation.ipynb` houses all of the code for generating temp files, training models, and evaluating models.

Be sure to change the data directory in the notebook to point to your data folder with all of the fasta files. 

Here is performance data for the models:

Model File| Pearson Coeff. |  Spearman Coeff.
------------ | ------------ | -------------
`ConvNet-1001-ABE-1603181216.h5`|0.5154 |0.3623
`ConvNet-101-ABE-1603322470.h5`| 0.4858|0.3421
`ConvNet-21-ABE-1603326140.h5`|  0.3819|0.2813
`ConvNet-101-CBE-1601246043.h5`|0.8003| 0.7639
`ResNet-1001-ABE-1601244091.h5`|0.5183|0.3637
