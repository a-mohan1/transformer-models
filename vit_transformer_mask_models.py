"""
Classification models based on time series transformer and vision transformer models
Author: Aruna Mohan
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import scipy
import itertools
import matplotlib.pyplot as plt

# TRANSFORMER AND VISION TRANSFORMER MODELS WITH MASKING

# TRANSFORMER MODEL
# Transformer model
class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, k):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = k
        
    def build(self, input_shape):
        # #Time dimensions (e.g., hours, min, sec): 
        time_dim = input_shape[-1]
        # Linear nonperiodic component
        self.wl = self.add_weight(name= 'wl', shape=(1,time_dim,1), initializer='uniform', trainable=True)
        self.bl = self.add_weight(name= 'bl', shape=(1,time_dim,1), initializer='uniform', trainable=True)
        # Sinusoidal periodic components
        self.ws = self.add_weight(name= 'ws', shape=(1,time_dim,self.k), initializer='uniform', trainable=True)
        self.bs = self.add_weight(name= 'bs', shape=(1,time_dim,self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)
        
    def call(self, inputs):
        # shape of inputs = (batch size, # time dimensions) for a single time step
        inputs = tf.cast(inputs, self.ws.dtype)
        inputs = tf.expand_dims(inputs, -1)
        # shape of inputs = (batch size, # time dimensions, 1)
        lin_inp = tf.math.multiply(inputs, self.wl) + self.bl
        # lin_inp = inputs * self.wl + self.bl # also works
        periodic_inp = tf.math.multiply(tf.tile(inputs, multiples=[1,1,self.k]), self.ws) + self.bs
        # periodic_inp = inputs * self.ws + self.bs # also works 
        periodic_act = tf.math.sin(periodic_inp)
                
        outputs = tf.keras.layers.Concatenate(axis=-1)([lin_inp, periodic_act])
        return outputs # shape = inputs.shape[0], inputs.shape[1], k+1
    
class AttentionBlock(tf.keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, key_dim=64, ff_dim=32, dropout=0, **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        # Attention 
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, **kwargs)
        self.add = tf.keras.layers.Add()
        self.attention_norm = tf.keras.layers.LayerNormalization()
        
        # Feed forward layers, first FF layer
        self.ff1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.ff_dropout = tf.keras.layers.Dropout(rate=dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization()
        
    def build(self, input_shape):
        # 2nd FF layer, Returns output with same shape as input 
        self.ff2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1, activation='relu')
        
    def call(self, inputs, mask=None):
        # Self-attention, inputs are both query and value
        x = self.attention(inputs, inputs, return_attention_scores=False, attention_mask=mask) 
        x = self.attention_norm(self.add([inputs, x]))
        x_ff = self.ff1(x)
        x_ff = self.ff2(x_ff)
        x_ff = self.ff_dropout(x_ff)
        outputs = self.ff_norm(self.add([x, x_ff]))
        
        return outputs
    
    def get_attention_scores(self, inputs, mask=None):
        _, scores = self.attention(inputs, inputs, training=False, return_attention_scores=True, attention_mask=mask) 
        return scores
    
class AttClassifier(tf.keras.Model):
    def __init__(self, name='AttClassifier', time2vec_dim=2, num_heads=2, key_dim=64, ff_dim=32, 
                num_attlayers=1, dropout=0, n_dense=128, num_classes=2, use_mask=False, **kwargs):
        super(AttClassifier, self).__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(k=time2vec_dim)
        self.attention_layers = [AttentionBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, 
                                                dropout=dropout) for _ in range(num_attlayers)]
        self.dense_layer = tf.keras.layers.Dense(units=n_dense, activation='relu')
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)
        self.classifier_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')
        self.use_mask = use_mask
        
    def call(self, inputs):
        # signal and time inputs shape: batch size, # time steps, # time dimensions
        if self.use_mask:
            time_inputs, signal_inputs, mask = inputs
        else:
            time_inputs, signal_inputs = inputs
            mask = None
        time_embedding = tf.keras.layers.TimeDistributed(self.time2vec)(time_inputs)
        # time embedding shape: batch size, # time steps, # time dimensions, k+1
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(signal_inputs, -1), time_embedding])
        # x shape: batch size, # time steps, # time dimensions, k+2
        
        for attention_layer in self.attention_layers:
            x = attention_layer(x, mask=mask)
        
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_layer(x)
        x = self.dropout_layer(x)
        outputs = self.classifier_layer(x)
        
        return outputs
    
    def get_last_attention_scores(self, inputs):
        # signal and time inputs shape: batch size, # time steps, # time dimensions
        if self.use_mask:
            time_inputs, signal_inputs, mask = inputs
        else:
            time_inputs, signal_inputs = inputs
            mask = None
        time_embedding = tf.keras.layers.TimeDistributed(self.time2vec)(time_inputs, training=False)
        # time embedding shape: batch size, # time steps, # time dimensions, k+1
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(signal_inputs, -1), time_embedding])
        # x shape: batch size, # time steps, # time dimensions, k+2
        
        # Run though all but last attention layer
        for attention_layer in self.attention_layers[:-1]:
            x = attention_layer(x, mask=mask, training=False)
        # Attention scores from last layer
        scores = self.attention_layers[-1].get_attention_scores(x, mask=mask)
        return scores
    
# Calculate attention mask
def calc_attention_mask(signal_lengths, max_length=1500):
    # Mask
    mask = [np.ones(siglen) for siglen in signal_lengths]
    mask = tf.keras.preprocessing.sequence.pad_sequences(mask, maxlen=max_length, padding='post')
    # https://github.com/keras-team/keras/issues/16248
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    # https://stackoverflow.com/questions/67805117/multiheadattention-attention-mask-keras-tensorflow-example
    # Dimensions: batch size, # attention heads, target sequence, 1, source sequence, 1
    # Extra dimensions of 1 are because Time2Vec embedding introduces an extra dimension
    mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    mask = mask.astype('bool')
    return mask

def create_transformer_classifier_1lead(max_length=1500, num_classes=2, time2vec_dim=8, 
                                        num_heads=2, key_dim=16, ff_dim=32, 
                                        num_attlayers=1, dropout=0.2, n_dense=128, use_mask=False):
    
    time_inputs = tf.keras.layers.Input(shape=(max_length,1))
    signal_inputs = tf.keras.layers.Input(shape=(max_length,1))
    if not use_mask:
        inputs = [time_inputs, signal_inputs]
    else:
        mask = tf.keras.layers.Input(shape=(1, 1, 1, max_length, 1))
        inputs = [time_inputs, signal_inputs, mask]
        
    outputs = AttClassifier(time2vec_dim=time2vec_dim, num_heads=num_heads, key_dim=key_dim,
                            ff_dim=ff_dim, num_attlayers=num_attlayers, dropout=dropout, 
                            n_dense=n_dense, num_classes=num_classes, use_mask=use_mask)(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attn_1lead") 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                        clipvalue=0.5
                                        )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def create_transformer_classifier_2leads(max_length=1500, num_classes=2, time2vec_dim=8, 
                                        num_heads=2, key_dim=16, ff_dim=32, 
                                        num_attlayers=1, dropout=0.2, n_dense=128, use_mask=False):

    # Time inputs
    time_inputs = tf.keras.layers.Input(shape=(max_length,2))
    # Main signal 
    main_signal = tf.keras.layers.Input(shape=(max_length, 1))  
    # Secondary signal
    sec_signal = tf.keras.layers.Input(shape=(max_length, 1))  
    # Signal inputs
    signal_inputs = tf.keras.layers.Concatenate(axis=-1)([main_signal, sec_signal])
    if not use_mask:
        inputs = [time_inputs, signal_inputs]
    else:
        mask = tf.keras.layers.Input(shape=(1, 1, 1, max_length, 1))
        inputs = [time_inputs, signal_inputs, mask]
    
    outputs = AttClassifier(time2vec_dim=time2vec_dim, num_heads=num_heads, key_dim=key_dim,
                            ff_dim=ff_dim, num_attlayers=num_attlayers, dropout=dropout, 
                            n_dense=n_dense, num_classes=num_classes, use_mask=use_mask)(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attn_2leads") 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                        clipvalue=0.5
                                        )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# ATTENTION MODEL WITH CLS TOKEN
# Attention classifier with cls token
@keras.saving.register_keras_serializable(package='AttnClassifier_clstoken')
class AttClassifier_clstoken(tf.keras.layers.Layer):
    def __init__(self, name='AttClassifier_clstoken', time2vec_dim=2, num_heads=2, key_dim=64, ff_dim=32, 
                num_attlayers=1, dropout=0, n_dense=128, num_classes=2, use_mask=False, **kwargs):
        super(AttClassifier_clstoken, self).__init__(name=name, **kwargs)
        self.time2vec_dim = time2vec_dim
        self.time2vec = Time2Vec(k=time2vec_dim)
        self.attention_layers = [AttentionBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, 
                                                dropout=dropout) for _ in range(num_attlayers)]
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.gelu)
        self.final_dense_layer = tf.keras.layers.Dense(units=time2vec_dim+2, activation=tf.nn.gelu)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)
        self.classifier_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')
        self.use_mask = use_mask
        
    def build(self, input_shape):
        # # time dimensions:
        time_dim = input_shape[0][-1]
        # Classifier token
        self.cls_token = self.add_weight(name='cls_token', shape=(1,1,time_dim,self.time2vec_dim+2), initializer='uniform', trainable=True)
        
    def call(self, inputs):
        # signal and time inputs shape: batch size, # time steps, # time dimensions
        if self.use_mask:
            time_inputs, signal_inputs, mask = inputs
        else:
            time_inputs, signal_inputs = inputs
            mask = None
        time_embedding = tf.keras.layers.TimeDistributed(self.time2vec)(time_inputs)
        # time embedding shape: batch size, # time steps, # time dimensions, k+1
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(signal_inputs, -1), time_embedding])
        # x shape: batch size, # time steps, # time dimensions, k+2
        
        # Classifier token
        cls_token = tf.repeat(self.cls_token, repeats=tf.shape(signal_inputs)[0], axis=0)
        x = tf.keras.layers.Concatenate(axis=1)([cls_token, x])
        # x shape: batch size, 1 + # time steps, # time dimensions, k+2
        
        for attention_layer in self.attention_layers:
            x = attention_layer(x, mask=mask)
        x = self.dense_layer(x)
        #x = self.dropout_layer(x)
        x = self.final_dense_layer(x)
        
        # Classifier token
        cls_x = x[:,0]
        # Flatten time dimensions and time encoding dimensions
        cls_x = self.flatten_layer(cls_x)
        
        outputs = self.classifier_layer(cls_x)
        
        return outputs
    
    def get_last_attention_scores(self, inputs):
        # signal and time inputs shape: batch size, # time steps, # time dimensions
        if self.use_mask:
            time_inputs, signal_inputs, mask = inputs
        else:
            time_inputs, signal_inputs = inputs
            mask = None
        time_embedding = tf.keras.layers.TimeDistributed(self.time2vec)(time_inputs, training=False)
        # time embedding shape: batch size, # time steps, # time dimensions, k+1
        x = tf.keras.layers.Concatenate(axis=-1)([tf.expand_dims(signal_inputs, -1), time_embedding])
        # x shape: batch size, # time steps, # time dimensions, k+2
        # Classifier token
        cls_token = tf.repeat(self.cls_token, repeats=tf.shape(signal_inputs)[0], axis=0)
        x = tf.keras.layers.Concatenate(axis=1)([cls_token, x])
        # x shape: batch size, 1 + # time steps, # time dimensions, k+2
        
        # Run though all but last attention layer
        for attention_layer in self.attention_layers[:-1]:
            x = attention_layer(x, mask=mask, training=False)
        # Attention scores from last layer
        scores = self.attention_layers[-1].get_attention_scores(x, mask=mask)
        return scores

# Calculate attention mask, including cls token
def calc_attention_mask_clstoken(signal_lengths, max_length=1500):
    # Mask, including signal length + cls token
    mask = [np.ones(siglen+1) for siglen in signal_lengths]
    mask = tf.keras.preprocessing.sequence.pad_sequences(mask, maxlen=max_length+1, padding='post')
    # https://github.com/keras-team/keras/issues/16248
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    # https://stackoverflow.com/questions/67805117/multiheadattention-attention-mask-keras-tensorflow-example
    # Dimensions: batch size, # attention heads, target sequence, 1, source sequence, 1
    # Extra dimensions of 1 are because Time2Vec embedding introduces an extra dimension
    mask = mask[:, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    mask = mask.astype('bool')
    return mask

def create_transformer_clstoken_classifier_1lead(max_length=1500, num_classes=2, time2vec_dim=8,
                                                 num_heads=2, key_dim=16, ff_dim=32, 
                                                 num_attlayers=1, dropout=0.2, n_dense=128, use_mask=False):

    time_inputs = tf.keras.layers.Input(shape=(max_length,1))
    signal_inputs = tf.keras.layers.Input(shape=(max_length,1))
    if not use_mask:
        inputs = [time_inputs, signal_inputs]
    else:
        mask = tf.keras.layers.Input(shape=(1, 1, 1, max_length+1, 1))
        inputs = [time_inputs, signal_inputs, mask]

    outputs = AttClassifier_clstoken(time2vec_dim=time2vec_dim, num_heads=num_heads, key_dim=key_dim,
                            ff_dim=ff_dim, num_attlayers=num_attlayers, dropout=dropout, 
                            n_dense=n_dense, num_classes=num_classes, use_mask=use_mask)(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attn_clstoken_1lead") 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                        clipvalue=0.5
                                        )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def create_transformer_clstoken_classifier_2leads(max_length=1500, num_classes=2, time2vec_dim=8,
                                                 num_heads=2, key_dim=16, ff_dim=32, 
                                                 num_attlayers=1, dropout=0.2, n_dense=128, use_mask=False):

    # Time inputs
    time_inputs = tf.keras.layers.Input(shape=(max_length,2))
    # Main signal 
    main_signal = tf.keras.layers.Input(shape=(max_length, 1))  
    # Secondary signal
    sec_signal = tf.keras.layers.Input(shape=(max_length, 1))  
    # Signal inputs
    signal_inputs = tf.keras.layers.Concatenate(axis=-1)([main_signal, sec_signal])
    if not use_mask:
        inputs = [time_inputs, signal_inputs]
    else:
        mask = tf.keras.layers.Input(shape=(1, 1, 1, max_length+1, 1))
        inputs = [time_inputs, signal_inputs, mask]
  
    outputs = AttClassifier_clstoken(time2vec_dim=time2vec_dim, num_heads=num_heads, key_dim=key_dim,
                            ff_dim=ff_dim, num_attlayers=num_attlayers, dropout=dropout, 
                            n_dense=n_dense, num_classes=num_classes, use_mask=use_mask)(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attn_clstoken_2leads") 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                        clipvalue=0.5
                                        )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# VISION TRANSFORMER MODEL
class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, embedding_dim, dropout=0.0, **kwargs):
        super(ViTEmbeddings, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patch_embedding_layer = tf.keras.layers.Conv1D(filters=embedding_dim, kernel_size=patch_size, strides=patch_size)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)
        
    def build(self, input_shape):
        # input shape = batch size, # time steps (image size), # time dimension (# channels)
        self.cls_token = self.add_weight(name="cls_token", shape=(1,1,self.embedding_dim), initializer="uniform", trainable=True)
        num_patches = input_shape[1]//self.patch_size
        self.position_embeddings = self.add_weight(name="positional_embeddings", 
                                                   shape=(1,num_patches+1,self.embedding_dim), 
                                                   initializer="uniform",
                                                   trainable=True
                                                  )
        
        '''
        # Alternative way of positional encoding
        position_embedding_layer = tf.keras.layers.Embedding(input_dim=num_patches+1, output_dim=self.embedding_dim, input_length=num_patches+1)
        positions_incl_cls_token = tf.range(start=0, limit=num_patches+1, delta=1)
        self.position_embeddings = position_embedding_layer(positions_incl_cls_token)
        '''
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        patch_embeddings = self.patch_embedding_layer(inputs)
        # patch embeddings: batch size, # patches, embedding_dim
        cls_tokens = tf.repeat(self.cls_token, repeats=input_shape[0], axis=0)
        # cls_tokens: batch_size, 1, embedding_dim
        # Concatenate cls_tokens with patch embeddings
        patch_embeddings = tf.concat((cls_tokens, patch_embeddings), axis=1)
        # patch embeddings = batch_size, 1 + # patches, embedding_dim
        
        # Add positional embedding to patch embeddings
        embeddings = patch_embeddings + self.position_embeddings
        embeddings = self.dropout_layer(embeddings)
        # embeddings = batch_size, 1 + # patches, embedding_dim
        return embeddings
        
class MLP(tf.keras.layers.Layer):
    # mlp_units: list of units in Dense layers
    def __init__(self, mlp_units, dropout, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout)
        self.dense_layers = [tf.keras.layers.Dense(units=units, activation=tf.nn.gelu) for units in mlp_units]
        
    def build(self, input_shape):
        self.final_dense_layer = tf.keras.layers.Dense(units=input_shape[-1], activation=tf.nn.gelu)
        
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
            x = self.dropout_layer(x)
        # Final Dense layer
        x = self.final_dense_layer(x)
        x = self.dropout_layer(x)
        return x
    
class ViT_AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_units, num_heads, key_dim, dropout=0.0, **kwargs):
        # key dim is same as embedding dim
        super(ViT_AttentionBlock, self).__init__(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.attn_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, **kwargs)
        self.add = tf.keras.layers.Add()
        self.mlp_layers = MLP(mlp_units=mlp_units, dropout=dropout)
        
    def call(self, inputs, mask=None):
        x = self.layernorm(inputs)
        # Self-attention
        x = self.attn_layer(x, x, attention_mask=mask, return_attention_scores=False)
        x = self.add([inputs, x])
        x_mlp = self.layernorm(x)
        x_mlp = self.mlp_layers(x_mlp)
        x = self.add([x, x_mlp])
        return x
    
    def get_attention_scores(self, inputs, mask=None):
        x = self.layernorm(inputs, training=False)
        _, scores = self.attn_layer(x, x, attention_mask=mask, training=False, return_attention_scores=True)
        return scores
    
@keras.saving.register_keras_serializable(package='ViTClassifier')
class ViTClassifier(tf.keras.layers.Layer):
    def __init__(self, num_classes=2, patch_size=30, num_att_layers=1, mlp_units=[128],
                 num_heads=2, embedding_dim=10, dropout=0.0, max_length=1500, use_mask=False,
                 **kwargs):
        super(ViTClassifier, self).__init__(**kwargs)
        self.vit_embeddings = ViTEmbeddings(patch_size, embedding_dim, dropout)
        self.vit_attention_layers = [ViT_AttentionBlock(mlp_units, num_heads, embedding_dim, 
                                                 dropout) for _ in range(num_att_layers)]
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.mlp_layers = MLP(mlp_units=mlp_units, dropout=dropout)
        self.classifier_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')
        self.use_mask = use_mask
            
    def call(self, inputs):
        if self.use_mask:
            signal_inputs, mask = inputs
        else:
            signal_inputs = inputs
            mask = None
            
        # Patch + positional embeddings
        x = self.vit_embeddings(signal_inputs)
        # Self-attention layers
        for att_layer in self.vit_attention_layers:
            x = att_layer(x, mask=mask)
        x = self.layer_norm(x)
        x = self.mlp_layers(x)
        # Classifier token
        cls_x = x[:,0]
        output = self.classifier_layer(cls_x)
        
        return output
    
    def get_last_selfattention(self, inputs):
        if self.use_mask:
            signal_inputs, mask = inputs
        else:
            signal_inputs = inputs
            mask = None
        
        x = self.vit_embeddings(signal_inputs, training=False)
        # Run through all but last layer
        for att_layer in self.vit_attention_layers[:-1]:
            x = att_layer(x, mask=mask, training=False)
        # Attention scores from last layer
        scores = self.vit_attention_layers[-1].get_attention_scores(x, mask=mask)
        return scores
    
# Calculate ViT mask
def calc_vit_mask(signal_lengths, patch_size=30, max_length=1500):
    # Mask
    # Nonzero mask in patches where there is at least one nonzero value, +1 for cls token
    nonzero_patchlengths = [-(siglen // -patch_size) for siglen in signal_lengths]
    mask = [np.ones(patchlen+1) for patchlen in nonzero_patchlengths]
    # Includes patches + classifier token
    total_num = max_length//patch_size + 1
    mask = tf.keras.preprocessing.sequence.pad_sequences(mask, maxlen=total_num, padding='post')
    # https://github.com/keras-team/keras/issues/16248
    mask = mask[:, tf.newaxis]
    mask = mask.astype('bool')
    return mask

def create_vit_classifier_1lead(max_length=1500, num_classes=2, patch_size=30, 
                                num_att_layers=1, mlp_units=[128], num_heads=2, 
                                embedding_dim=10, dropout=0.0, use_mask=False):
    
    signal_inputs = tf.keras.layers.Input(shape=(max_length, 1))
    if use_mask:
        # Mask size includes patches + classifier token
        mask_size = max_length//patch_size + 1
        mask = tf.keras.layers.Input(shape=(1, mask_size))
        inputs = [signal_inputs, mask]
    else:
        inputs = signal_inputs
    
    outputs = ViTClassifier(num_classes=num_classes, patch_size=patch_size, 
                            num_att_layers=num_att_layers, mlp_units=mlp_units, 
                            num_heads=num_heads, embedding_dim=embedding_dim, 
                            dropout=dropout, max_length=max_length, use_mask=use_mask)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_1lead")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                    clipvalue=0.5
                                    )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def create_vit_classifier_2leads(max_length=1500, num_classes=2, patch_size=30, 
                                num_att_layers=1, mlp_units=[128], num_heads=2, 
                                embedding_dim=10, dropout=0.0, use_mask=False):

    # Main signal 
    main_signal = Input(shape=(max_length, 1))  
    # Secondary signal
    sec_signal = Input(shape=(max_length, 1))  
    # Inputs to VIT, shape=batch size, # time steps, # time dimensions=2 
    signal_inputs = tf.keras.layers.Concatenate(axis=-1)([main_signal, sec_signal])
    if use_mask:
        # Mask size includes patches + classifier token
        mask_size = max_length//patch_size + 1
        mask = tf.keras.layers.Input(shape=(1, mask_size))
        inputs = [signal_inputs, mask]
    else:
        inputs = signal_inputs
    
    outputs = ViTClassifier(num_classes=num_classes, patch_size=patch_size, 
                            num_att_layers=num_att_layers, mlp_units=mlp_units, 
                            num_heads=num_heads, embedding_dim=embedding_dim, 
                            dropout=dropout, max_length=max_length, use_mask=use_mask)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_2leads")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                         clipvalue=0.5
                                        )
    # Compile
    loss = 'binary_crossentropy' if num_classes==2 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# LEARNING RATE SCHEDULER
# Learning rate scheduler
def lr_scheduler(epoch, warmup_epochs=15, decay_epochs=100, initial_lr=1e-4, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        return (base_lr - initial_lr)*epoch/warmup_epochs + initial_lr
    elif epoch > warmup_epochs and epoch <= warmup_epochs + decay_epochs:
        return (base_lr - min_lr)*(1-(epoch-warmup_epochs)/decay_epochs) + min_lr
    else:
        return min_lr


# ATTENTION HEATMAPS PLOTTING
# Function to return plots of attention maps superimposed on signals
def plot_attn_signal(df_test: pd.DataFrame, attention_scores: tf.Tensor, max_length: int, class_labels: np.array) -> plt.subplots:
    # Interpolate signals and attention maps to same length (without zero padding)

    # Signal lengths
    signal_lengths = [len(df_test['main_signal'].values[i]) for i in range(len(df_test))]
    df_test['signal_length'] = signal_lengths
    # Interpolate all signal lengths to max length
    df_test['main_signal_interp'] = df_test['main_signal'].apply(lambda y: scipy.interpolate.interp1d(np.linspace(0, 1, len(y)), y)(np.linspace(0,1,max_length)))

    # Number of classes
    num_classes = df_test['predicted label encoded'].nunique()
    # Number of attention heads
    num_attn_heads = int(tf.shape(attention_scores)[1])
    # Interpolate attention maps, and sum and normalize
    # Success cases
    attention_norm = np.zeros((num_classes, max_length, num_attn_heads))
    # Failure cases
    attention_fail_norm = np.zeros((num_classes, max_length, num_attn_heads))
    for i in range(len(df_test)):
        # All (1st and 2nd) attention heads
        attn = attention_scores[i, :, 0, 1:]
        attn = tf.transpose(attn)
        attn = tf.expand_dims(tf.expand_dims(attn, 0), 0)
        attn = tf.image.resize(attn, (1, max_length))[0,0] # for Vit; tf.image.resize works on 2 middle dims
        # Take attention values for signal length, excluding zero-padding at the end
        signal_length = df_test.loc[i, 'signal_length']
        attn_siglen = attn[:-(max_length-signal_length), :]
        # Interpolate attention values to max signal length
        attn_interp = scipy.interpolate.interp1d(np.linspace(0, 1, len(attn_siglen)), attn_siglen, axis=0)(np.linspace(0,1,max_length))
        # Sum by label (correctly predicted or wrongly predicted)
        pred_label_i = df_test.loc[i, 'predicted label encoded']
        true_label_i = df_test.loc[i, 'label_encoded']
        if pred_label_i == true_label_i:
            attention_norm[pred_label_i,:,:] += attn_interp
        else:
            attention_fail_norm[true_label_i,:,:] += attn_interp

    # Mean and std deviation of signal for test samples
    # Correctly predicted
    df_test_success = df_test[df_test['label_encoded']==df_test['predicted label encoded']].reset_index()
    RRR_main_signal_interp_mean = df_test_success.groupby('label_encoded')['main_signal_interp'].mean()
    RRR_main_signal_interp_std = [df_test_success[df_test_success['label_encoded']==i]['main_signal_interp'].values.std() for i in df_test_success['label_encoded'].unique()]
    # Wrongly predicted
    df_test_fail = df_test[df_test['label_encoded']!=df_test['predicted label encoded']].reset_index()
    fail_RRR_main_signal_interp_mean = df_test_fail.groupby('label_encoded')['main_signal_interp'].mean()
    fail_RRR_main_signal_interp_std = [df_test_fail[df_test_fail['label_encoded']==i]['main_signal_interp'].values.std() for i in df_test_fail['label_encoded'].unique()]    
    
    # Plots
    
    # Success cases
    figs_c, axs = plt.subplots(num_attn_heads*num_classes, figsize=(5,15))
    plt.tight_layout(h_pad=5, w_pad=None)
    for ifig in range(num_attn_heads*num_classes):
        i_attn_head = int(ifig/num_classes)
        i_class = ifig%num_classes
        # Normalize
        attention_norm[i_class,:,i_attn_head] = (attention_norm[i_class,:,i_attn_head]-min(attention_norm[i_class,:,i_attn_head]))/(max(attention_norm[i_class,:,i_attn_head]) - min(attention_norm[i_class,:,i_attn_head]))
        # Plot
        img = axs[ifig].imshow(tf.reshape(attention_norm[i_class,:,i_attn_head], (1,-1)), cmap='Reds', aspect="auto", extent=(0,max_length,-300,800))
        linestyle = {'linewidth': 1}
        axs[ifig].errorbar(np.arange(max_length), RRR_main_signal_interp_mean[i_class], yerr=RRR_main_signal_interp_std[i_class], ecolor='gray', color='b', **linestyle)
        axs[ifig].set_title(f"Class # {class_labels[i_class]}, Attention head # {i_attn_head}")
        plt.colorbar(img)
        
    # Failure cases
    figs_w, axs = plt.subplots(num_attn_heads*num_classes, figsize=(5,15))
    plt.tight_layout(h_pad=5, w_pad=None)
    for ifig in range(num_attn_heads*num_classes):
        i_attn_head = int(ifig/num_classes)
        i_class = ifig%num_classes
        # Normalize
        attention_fail_norm[i_class,:,i_attn_head] = (attention_fail_norm[i_class,:,i_attn_head]-min(attention_fail_norm[i_class,:,i_attn_head]))/(max(attention_fail_norm[i_class,:,i_attn_head]) - min(attention_fail_norm[i_class,:,i_attn_head]))
        # Plot
        img = axs[ifig].imshow(tf.reshape(attention_fail_norm[i_class,:,i_attn_head], (1,-1)), cmap='Reds', aspect="auto", extent=(0,max_length,-300,800))
        linestyle = {'linewidth': 1}
        axs[ifig].errorbar(np.arange(max_length), fail_RRR_main_signal_interp_mean[i_class], yerr=fail_RRR_main_signal_interp_std[i_class], ecolor='gray', color='b', **linestyle)
        axs[ifig].set_title(f"Class # {class_labels[i_class]}, Attention head # {i_attn_head}")
        plt.colorbar(img)
    
    # Success cases
    for i_class, i_attn_head in itertools.product(range(num_classes), range(num_attn_heads)):
        # Save figures to files
        fig, ax = plt.subplots(figsize=(4,2))
        img = ax.imshow(tf.reshape(attention_norm[i_class,:,i_attn_head], (1,-1)), cmap='Reds', aspect="auto", extent=(0,max_length,-300,800))
        linestyle = {'linewidth': 1}
        ax.errorbar(np.arange(max_length), RRR_main_signal_interp_mean[i_class], yerr=RRR_main_signal_interp_std[i_class], ecolor='gray', color='b', **linestyle)
        bar = plt.colorbar(img)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.ylim([-300,800])
        img_name = f"correct_cls_{class_labels[i_class]}_atthd_{i_attn_head}.png"
        fig.savefig(img_name, bbox_inches='tight', dpi=1200)
        #img_name = f"correct_cls_{class_labels[i_class]}_atthd_{i_attn_head}.eps"
        #fig.savefig(img_name, bbox_inches='tight', format='eps', dpi=1200)
        plt.show()
    # Failure cases
    for i_class, i_attn_head in itertools.product(range(num_classes), range(num_attn_heads)):
        # Save figures to files
        fig, ax = plt.subplots(figsize=(4,2))
        img = ax.imshow(tf.reshape(attention_fail_norm[i_class,:,i_attn_head], (1,-1)), cmap='Reds', aspect="auto", extent=(0,max_length,-300,800))
        linestyle = {'linewidth': 1}
        ax.errorbar(np.arange(max_length), fail_RRR_main_signal_interp_mean[i_class], yerr=fail_RRR_main_signal_interp_std[i_class], ecolor='gray', color='b', **linestyle)
        bar = plt.colorbar(img)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.ylim([-300,800])
        img_name = f"fail_cls_{class_labels[i_class]}_atthd_{i_attn_head}.png"
        fig.savefig(img_name, bbox_inches='tight', dpi=1200)
        #img_name = f"fail_cls_{class_labels[i_class]}_atthd_{i_attn_head}.eps"
        #fig.savefig(img_name, bbox_inches='tight', format='eps', dpi=1200)
        plt.show()
        
    return figs_c, figs_w
