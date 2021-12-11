# Bussiness Case
## Brain tumor classification from MRI Images using Convolution Neural Network

Brain tumour is a very serious brain cancer. It is present or become due to the separation of the brain cells. In the recent field of this study, tells us that deep learning will help in health industry of medical diseases imaging in the Medical Diagnostic of all the diseases. CNN is mostly used in this Machine learning algorithm. Likewise, in this project also, we bring out the Convolution Neural Network algorithm, image processing and data augmentation to say the brain images are cancerous and which are not cancerous. This project will require less computational power due to the transfer learning compared to the old CNN model.

# Methodology

![image](https://user-images.githubusercontent.com/69073063/145675584-5af474d8-b8a6-4f0e-a756-4d0384750abc.png)

# Dataset

The Dataset used in this project contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous. 

Dataset can be downloaded from here, https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

![image](https://user-images.githubusercontent.com/69073063/145676221-78b5bb27-27ab-443b-9d4b-50a21bc91a49.png)

![image](https://user-images.githubusercontent.com/69073063/145676235-85ad9bdf-59fd-41e0-9744-f6c694cfa95b.png)

# Data Split:
The data was split in the following way:

60% of the data for training.

20% of the data for validation.

20% of the data for testing.

![image](https://user-images.githubusercontent.com/69073063/145675174-2f9fdc0b-9a41-48a3-b871-4a8364240152.png)

# Data Augmentation

Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in tackling the data imbalance issue in the data.

![image](https://user-images.githubusercontent.com/69073063/145676256-030f9e84-3aed-446e-8c32-af0cdd66b2be.png)

# Transfer learning

Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

![image](https://user-images.githubusercontent.com/69073063/145675358-82029b4e-721f-4bf2-9b9b-3588caa17375.png)

**Using the transfer learning approach we compared the performance of our scratched CNN model with pre-trained Xception, VGG-16, VGG-19, and ResNet-50 models.**

# Neural Network Architecture

![Architecture-of-Vgg16-A-Vgg19-B-and-ResNet-C](https://user-images.githubusercontent.com/69073063/145675526-f9e1200b-c2a7-4f9f-bf21-a35cac86cdfc.png)

Architecture of VGG16 (A), VGG19 (B), and ResNet (C).

# Model Summary 

![image](https://user-images.githubusercontent.com/69073063/145675616-c7645747-1d44-40bf-bfa3-cd117d6b71cf.png)

![image](https://user-images.githubusercontent.com/69073063/145676297-cb455e9a-1ca2-4bf6-9063-3533c81f8aae.png)

# Model Deployment 

## Commands to run the project locally

```scala
activate conda py38
docker build -t vgg16-model .
docker run -it --rm -p 8080:8080 vgg16-model:latest
python3 predict_local.py
``` 
![image](https://user-images.githubusercontent.com/69073063/145675864-8b6a4ec7-ff75-473b-8895-957616bdeed1.png)

![image](https://user-images.githubusercontent.com/69073063/145675889-cea6ba7b-722a-49df-8adb-9b86504c5ec4.png)

## Commands to run the project from cloud service

**Service is already running on the cloud using AWS Lambda.**

![image](https://user-images.githubusercontent.com/69073063/145675988-ff68bd8d-da51-4578-86b0-83970d105370.png)

**Exposing the Lambda function using API Gateway**

![image](https://user-images.githubusercontent.com/69073063/145676021-c3a508ef-f147-46fc-aa70-99ce4707de9d.png)

**public endpoint that could be tested:** 

```scala
python3 predict_cloud.py
``` 

![image](https://user-images.githubusercontent.com/69073063/145675918-66005e69-8211-4a94-841e-871ff6371b04.png)


**I will terminate the service on 20.12.2021, after the end of peer-review week.**
