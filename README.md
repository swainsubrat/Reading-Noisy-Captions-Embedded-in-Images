# Reading Noisy Captions Embedded in Images
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
  - data/ \
       (this is inside the data folder)train_data/
  -  train.py
  -  constants.py
  -  etc....

Run:\
`python train.py`

Trained weight is present at: \
https://drive.google.com/drive/folders/1AdlhONGrZAayxKTqIrjMLOAGBy5i3F7x?usp=sharing \

Download and put it inside the data/ folder and run:\
`python testing.py`

Key architecture explained:
The backbone of the encoder is a convolutional network which could be any of VGG16, VGG19, ResNet-50, resNet-101 or anything you deem fit for the task at hand. For the explanation of the architecture, we take ResNet-101 as our backbone. Specially designed Convolutional networks like ResNet are really good at classification of images and they capture the salient features of the image. We have used pretrained ResNet-101 and fine tuned its last few layers. The output of the encoder is (batch size,14,14,2048), where Width=14, Height=14, No of channels=2048. The input images to the encoder could possible be of varying dimensions as we have used adaptive pool. But it will require computational overhead, hence we have resorted to resizing the inputs to 256 * 256.

The decoder consists of LSTM cell with attention. 

