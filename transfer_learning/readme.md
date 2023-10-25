# ENPM673  Final Project : Semantic Segmentation

## Student Details

|Name|Jay Prajapati|Akash Parmar| Sarin Mathew | Aditi Bhoir | Naveen Anil
|---|:---:|:---:|---|:---:|:---:|
|UID|119208625|118737430|119390382|119197257|119398593|
|Directory ID|jayp|akasparm|sarinann|abhoir|nvnanil|
|Email ID|jayp@umd.edu|akasparm@umd.edu|sarinann@umd.edu|abhoir@umd.edu|nvnanil@umd.edu|

## Dependencies used

|Modules|
|---|
|Python|3|
|Time|
|Numpy|
|Matplotlib|
|tqdm|
|tensorflow|
|os|
|torch|
|CUDA|


## Links

|Item|Link|
|---|:---:|
|Dataset|[here](https://drive.google.com/drive/folders/1KENOPSga_Rek4cmaOB7MErsrAR67Wsp7?usp=share_link)|


## Run Code

Download the zip folder ```transfer_learning.zip``` and extract it. 

Navigate to the ```transfer_leraning``` folder.

```sh
cd transfer_learning_dataset
```

Downlaod the ```transfer_learning_dataset.zip``` in the present directory and extract in the present directory

Edit the ```main.py``` file and give the path to parent directory as mentioned in the file.

Run the ```main.py``` file

```sh
python3 main.py
```
It will save the model as ```transfer_learning.pkl``` and will generate the plots for losses and accuracy.