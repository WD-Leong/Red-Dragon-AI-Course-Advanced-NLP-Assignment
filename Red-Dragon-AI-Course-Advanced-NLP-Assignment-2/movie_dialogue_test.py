import re
import time
import numpy as np
import pickle as pkl

import tensorflow as tf
import tf_ver2_transformer_modified as tf_transformer

# Model Parameters. #
seq_encode = 10
seq_decode = 11

num_heads   = 4
num_layers  = 3
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/TF_Models/transformer_seq2seq_v1"

tmp_pkl_file = tmp_path + "movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2word = pkl.load(tmp_file_load)
    word2idx = pkl.load(tmp_file_load)
vocab_size = len(word2idx)

SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = tf_transformer.TransformerNetwork(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, vocab_size, seq_encode, seq_decode, 
    embed_size=hidden_size, p_keep=prob_keep)
seq2seq_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optimizer=seq2seq_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Placeholders to store the batch data. #
tmp_test_in  = np.zeros([1, seq_encode], dtype=np.int32)
tmp_test_dec = SOS_token * np.ones([1, seq_decode], dtype=np.int32)

while True:
    tmp_phrase_in = input("Enter your phrase: ")
    if tmp_phrase_in == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token
        
        tmp_i_tok = re.sub(
            r"[^\w\s]", " ", tmp_phrase_in.lower()).split(" ")
        tmp_i_idx = [
            word2idx.get(x, UNK_token) for x in tmp_i_tok]
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        gen_ids = seq2seq_model.infer(tmp_test_in, tmp_test_dec)
        gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase = " ".join(gen_phrase)
        
        print("")
        print("Input Phrase:")
        print(" ".join([idx2word[x] for x in tmp_i_idx]))
        print("Generated Phrase:")
        print(gen_phrase)
        print("-" * 50)

        





