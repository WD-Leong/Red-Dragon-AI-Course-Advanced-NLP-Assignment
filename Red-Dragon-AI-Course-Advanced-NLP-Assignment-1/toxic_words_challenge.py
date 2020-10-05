# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:20:22 2020

@author: admin
"""

import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

# Define a Bi-Directional Transformer Model. #
def build_model(
    max_length, vocab_size, 
    embed_size=32, n_classes=6, tmp_pi=0.95):
    tmp_b  = tf.math.log((1.0-tmp_pi)/tmp_pi)
    x_bias = tf.Variable([tmp_b], name="x_bias")
    
    x_input = tf.keras.Input(
        shape=(max_length,), name='x_input')
    x_embed = tf.keras.layers.Embedding(
        vocab_size, embed_size)(x_input)
    
    x_conv1 = tf.keras.layers.Conv1D(
        embed_size*2, 5, strides=2)(x_embed)
    x_relu1 = tf.nn.relu(x_conv1)
    x_conv2 = tf.keras.layers.Conv1D(
        embed_size*4, 5, strides=2)(x_relu1)
    x_relu2 = tf.nn.relu(x_conv2)
    
    x_flatten = layers.Flatten()(x_relu2)
    x_linear1 = layers.Dense(
        128, activation="relu", name="linear1")(x_flatten)
    x_linear2 = layers.Dense(
        32, activation="relu", name="linear2")(x_linear1)
    x_logits  = layers.Dense(
        n_classes, activation="linear", name="logits")(x_linear2)
    x_outputs = x_logits + x_bias
    
    toxic_model = tf.keras.Model(
        inputs=x_input, outputs=x_outputs)
    return toxic_model

def model_loss(labels, logits, weights):
    loss = tf.multiply(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits), weights)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

def train_step(
    model, embed_in, labels, 
    optimizer, learning_rate=1.0e-3):
    optimizer.lr.assign(learning_rate)
    
    with tf.GradientTape() as grad_tape:
        tmp_weight = np.where(labels == 0, 1.0, 25.0)
        tmp_logits = model(embed_in, training=True)
        tmp_losses = model_loss(
            labels, tmp_logits, tmp_weight)
        
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model.trainable_variables)
        optimizer.apply_gradients(
            zip(tmp_gradients, model.trainable_variables))
    return tmp_losses

# Define the training. #
def train(
    model, train_dataset, batch_size, 
    epochs, optimizer, init_lr=1.0e-3, decay=0.75):
    train_losses = []
    
    tot_losses = 0.0
    for epoch in range(epochs):
        print("Epoch", str(epoch) + ":")
        start_time = time.time()
        learn_rate = decay**epoch * init_lr
        
        # Cool the GPU. #
        if (epoch+1) % 5 == 0:
            print("Cooling GPU.")
            #time.sleep(120)
        
        epoch_step = 0
        for tmp_embed, tmp_label in train_dataset:
            tmp_losses = train_step(
                model, tmp_embed, tmp_label, 
                optimizer, learning_rate=learn_rate)
            
            epoch_step += 1
            tot_losses += tmp_losses.numpy()
        
        avg_losses = tot_losses / epoch_step
        tot_losses = 0.0
        elapsed_time = (time.time() - start_time) / 60.0
        train_losses.append((epoch+1, avg_losses))
        
        epoch += 1
        print("Learning Rate:", str(optimizer.lr.numpy()))
        print("Elapsed time:", str(elapsed_time), "mins.")
        print("Average Epoch Loss:", str(avg_losses) + ".")
        print("-" * 75)
    
    tmp_train_df = pd.DataFrame(
        train_losses, columns=["epoch", "train_loss"])
    return tmp_train_df

# Parameters. #
n_epochs = 25
n_classes  = 6
embed_size = 32
batch_size = 256

# Load the data. #
print("Loading and processing the data.")

tmp_path = "C:/Users/admin/Desktop/Red Dragon/Advanced NLP/"
train_data = pd.read_csv(tmp_path + "toxic_words/train.csv")
test_data  = pd.read_csv(tmp_path + "toxic_words/test.csv")

train_data = train_data.fillna("blank")
test_data  = test_data.fillna("blank")
test_label = pd.read_csv(
    tmp_path + "toxic_words/test_labels.csv")
test_label = test_label.replace(
    to_replace=-1, value=0, inplace=False)

test_data = test_data.merge(test_label, on="id")
del test_label

# Form the vocabulary. #
train_corpus = list(train_data["comment_text"].values)
test_corpus  = list(test_data["comment_text"].values)

w_counter  = Counter()
max_length = 250
min_count  = 10
tmp_count  = 0
for tmp_comment in train_corpus:
    tmp_comment = tmp_comment.replace("\n", " \n ")
    tmp_tokens  = [
        x for x in word_tokenizer(tmp_comment.lower()) if x != ""]
    
    if len(tmp_tokens) > max_length:
        continue
    w_counter.update(tmp_tokens)
    
    tmp_count += 1
    if tmp_count % 25000 == 0:
        proportion = tmp_count / len(train_corpus)
        print(str(proportion*100)+"%", "comments processed.")

vocab_list = sorted(
    [x for x, y in w_counter.items() if y >= min_count])
vocab_list = sorted(vocab_list + ["EOS", "PAD", "UNK"])
vocab_size = len(vocab_list)

idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])
print("Vocabulary Size:", str(len(vocab_list)))

# Define the classifier model. #
toxic_model = build_model(
    max_length+1, vocab_size, embed_size=embed_size)
print(toxic_model.summary())

# Format the data before training. #
y_cols = ["toxic", "severe_toxic", "obscene", 
          "threat", "insult", "identity_hate"]
y_train = train_data[y_cols].values
y_test  = test_data[y_cols].values

PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]
EOS_token = word2idx["EOS"]

word_indices = PAD_token * np.ones(
    [len(y_train), max_length+1], dtype=np.int32)
for n in range(len(train_corpus)):
    tmp_comment = train_corpus[n].replace("\n", " \n ")
    tmp_tokens  = [
        x for x in word_tokenizer(tmp_comment.lower()) if x != ""]
    
    m = len(tmp_tokens)
    if m > max_length:
        m = max_length + 1
        tmp_tokens = tmp_tokens[:m]
    
    word_indices[n, :m] = [
        word2idx.get(x, UNK_token) for x in tmp_tokens]
    if m <= max_length:
        word_indices[n, m] = EOS_token
del train_corpus

test_indices = PAD_token * np.ones(
    [len(test_data), max_length+1], dtype=np.int32)
for n in range(len(test_data)):
    tmp_comment = test_corpus[n].replace("\n", " \n ")
    tmp_tokens  = [
        x for x in word_tokenizer(tmp_comment.lower()) if x != ""]
    
    m = len(tmp_tokens)
    if m > max_length:
        m = max_length + 1
        tmp_tokens = tmp_tokens[:m]
    
    test_indices[n, :m] = [
        word2idx.get(x, UNK_token) for x in tmp_tokens]
    if m <= max_length:
        test_indices[n, m] = EOS_token
del test_corpus

# Form the tensor dataset. #
train_tf_data = tf.data.Dataset.from_tensor_slices((
    tf.cast(word_indices, tf.int32), tf.cast(y_train, tf.float32)))
train_tf_data = train_tf_data.batch(batch_size, drop_remainder=True)

test_tf_data = tf.data.Dataset.from_tensor_slices(
    tf.cast(test_indices, tf.int32))
test_tf_data = \
    test_tf_data.batch(batch_size, drop_remainder=False)

# Train the model. #
print('Fit model on training data.')
optimizer = tf.keras.optimizers.Adam()
tmp_train_df = train(
    toxic_model, train_tf_data, batch_size, 
    n_epochs, optimizer, init_lr=0.01, decay=0.95)

# Run on the test data. #
pred_list = []
for test_mat in test_tf_data:
    pred_list.append(toxic_model(test_mat, training=False))
pred_probs = np.concatenate(tuple(pred_list), axis=0)
pred_label = np.where(pred_probs > 0.5, 1, 0)
del pred_list

num_correct = np.count_nonzero(pred_label == y_test)
num_labels  = pred_label.shape[0] * pred_label.shape[1]
confusion_mat = np.zeros([2, 2], dtype=np.float32)
for m in range(n_classes):
    tmp_true = y_test[:, m]
    tmp_pred = pred_probs[:, m]
    
    # True Negatives. #
    confusion_mat[0, 0] += len(
        [x for x in range(len(tmp_true)) \
         if tmp_true[x] == 0 and tmp_pred[x] < 0.5])
    
    # False Positives. #
    confusion_mat[0, 1] += len(
        [x for x in range(len(tmp_true)) \
         if tmp_true[x] == 0 and tmp_pred[x] >= 0.5])
    
    # False Negatives. #
    confusion_mat[1, 0] += len(
        [x for x in range(len(tmp_true)) \
         if tmp_true[x] == 1 and tmp_pred[x] < 0.5])
    
    # True Positives. #
    confusion_mat[1, 1] += len(
        [x for x in range(len(tmp_true)) \
         if tmp_true[x] == 1 and tmp_pred[x] >= 0.5])

# Save the files. #
tmp_train_df.to_csv(
    tmp_path + "toxic_word_train_loss.csv", index=False)

# Print the diagnostics. #
recall = \
    confusion_mat[1, 1] / (confusion_mat[1, 0] + confusion_mat[1, 1])
precision = \
    confusion_mat[1, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1])
print(confusion_mat)
print("Precision:", str(precision))
print("Recall:", str(recall))

# Plot the training loss. #
fig, ax = plt.subplots()
ax.plot(tmp_train_df["epoch"], tmp_train_df["train_loss"])
ax.set(xlabel="Epoch", ylabel="Training Loss")
fig.suptitle("Toxic Word Classification Training Loss")
fig.savefig(tmp_path + "training_loss.jpg", dpi=199)
plt.close("all")
del fig, ax
