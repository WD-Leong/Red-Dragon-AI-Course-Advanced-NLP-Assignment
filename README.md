# Red-Dragon-AI-Course-Advanced-NLP-Assignment-2

## Movie Dialogue Chatbot
We now move on to the 2nd assignment, which is an NLP project of our choice. This assignment trains a movie dialogue chatbot using a Transformer network. The pre-processing of the data follow this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. The data can be obtained [here](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

### 2.1 Transformer Model
Our Transformer model makes some modifications to the original model in that it trains a positional embedding layer at each layer of the encoder and decoder. In addition, it also adds a residual connection between the input embeddings at both the encoder and decoder. Apart from that, there were no further modifications made. The model uses 6 layers for both the encoder and decoder, a hidden size of 512 and a feed-forward size of 2048. The sequence length at both the encoder and decoder was set to 10, and a vocabulary of the most common 8000 words was used. A gradient clipping value of 1.0 was set during training as well. The model as returned by `seq2seq_model.summary()` is as follows:
```
Layer (type)                 Output Shape              Param #
=================================================================
Total params: 57,198,080
Trainable params: 57,198,080
Non-trainable params: 0
_________________________________________________________________
```
Due to limitations on the GPU card, we accumulate the gradients manually across sub-batches of 32, then average it to apply the overall weight update across a larger batch, since we observe that larger batch sizes tend to stabilise the training of Transformer networks. Following the [T5 paper](https://arxiv.org/abs/1910.10683), 2000 warmup steps with a constant learning rate was applied `step_val = float(max(n_iter+1, warmup_steps))**(-0.5)`.

### 2.2 Training the Dialogue Transformer Network
As the training progressed, the quality of the response was observed to get better and better.
```
--------------------------------------------------
Iteration 250:
Elapsed Time: 0.913459050655365 mins.
Average Loss: 27.461065521240233
Gradient Clip: 1.0
Learning Rate: 0.0009882117

Input Phrase:
i didn t know which side you were on
Generated Phrase:
i EOS EOS EOS EOS EOS EOS EOS EOS EOS PAD
Actual Response:
now you know
--------------------------------------------------
```
