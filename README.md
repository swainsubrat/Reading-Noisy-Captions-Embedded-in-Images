# urban-fiesta
The task is to predict the text embedded into the image (irrespective of the background image itself

Reproduce:

Clone repo: \
`git clone https://github.com/swainsubrat/urban-fiesta.git` \
`cd urban-fiesta`

Install Dependencies(python>=3.8 required): \
`pip install -r requirements.txt`

Download Data: \
https://drive.google.com/drive/folders/1HeLLaFvVJ3YctqxvHFRBlZnZZ8DntQBG?usp=sharing \

The link contains 2 folders(train_data, test_data) and a few files. Download only train_data folder(2gb). \
Make a folder inside the repo(Note you're inside the repo dir)
`mkdir data` \

put the train_data folder inside the data folder.

Hence, the folder structure would be:

urban-fiesta/
  - data/
       (this is inside the data folder)train_data/
  -  train.py
  -  constants.py
  -  etc....

Run:\
`python train.py`
