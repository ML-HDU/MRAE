# PyTorch Implementation of MRAE

This repository provides a PyTorch implementations of the matrix randomized autoencoder method presented in our paper 
<br>**"Matrix randomized autoencoder"**.
<br>You can find the original paper at <https://www.sciencedirect.com/science/article/pii/S0031320323006908>

## Citation

If you use our work, please also cite the paper:

	@article{ZHANG2024109992,
		title = {Matrix randomized autoencoder},
		journal = {Pattern Recognition},
		volume = {146},
		pages = {109992},
		year = {2024},
		issn = {0031-3203},
		doi = {https://doi.org/10.1016/j.patcog.2023.109992},
		url = {https://www.sciencedirect.com/science/article/pii/S0031320323006908},
		author = {Shichen Zhang and Tianlei Wang and Jiuwen Cao and Wandong Zhang and Badong Chen}
	}
	

## Runing Environment

This program uses some common Python packages including Numpy, PyTorch, Scipy, Sklearn, Matplotlib. These packages can be installed easily by the Official Website description. I suggest using "conda" to conduct installation.

## Instruction

The dataset with '.mat' format is used in this program. The dataset is divided into the following four files:

	'train_data.mat': training data saved in the variable 'P' in which each column is a sample.
	'train_label.mat': training label saved in the variable 'T' in which each row is a sample.
	'test_data.mat': testing data saved in the variable 'TV_P' in which each column is a sample.
	'test_label.mat':  testing label saved in the variable 'TV_T' in which each row is a sample.
We provide the MNIST and Fashion-MNIST as examples. For MNIST dataset, you should first download the original data files into the folder './datasets/MNIST/'. Then, the MATLAB program in "./datasets/MNIST/my_formatting.m" can help you derive the four files 'train_data.mat', 'train_label.mat', 'test_data.mat' and 'test_label.mat'.

The algorithms can be coducted by opening "my_main_with_fixed_parameters.py" and running it, where the hyper-parameters can be also set/modified in this files.

## License
MIT



