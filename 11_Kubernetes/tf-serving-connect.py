
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('autosave', '0')


# In[10]:


get_ipython().system('pip install grpcio==1.42.0 tensorflow-serving-api==2.6.2')


# In[11]:


get_ipython().system('pip install keras-image-helper')


# In[13]:


import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# In[14]:


host = 'localhost:8500'
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


# In[16]:


from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299, 299))


# In[17]:


url = "http://bit.ly/mlbookcamp-pants"
X = preprocessor.from_url(url)


# In[18]:


X.shape


# In[19]:


def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)


# In[20]:


pb_request = predict_pb2.PredictRequest()

pb_request.model_spec.name = 'clothing-model'
pb_request.model_spec.signature_name = 'serving_default'
pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X)) 


# In[21]:


pb_result = stub.Predict(pb_request, timeout=20.0)


# In[22]:


pred = pb_result.outputs['dense_7'].float_val


# In[23]:


labels = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
] 

result = {c: p for c, p in zip(labels, pred)}


# In[24]:


result

