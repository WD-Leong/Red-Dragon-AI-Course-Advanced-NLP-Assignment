# Red-Dragon-AI-Course-Advanced-NLP-Assignment-2

## Movie Dialogue Chatbot
We now move on to the 2nd assignment, which is an NLP project of our choice. This assignment trains a movie dialogue chatbot using a [Transformer](https://arxiv.org/abs/1706.03762) network. The pre-processing of the data follow this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. The data can be obtained [here](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

### Data Pre-processing
Similar to our Toxic Word challenge assignment, we apply simple pre-processing for the data, including lower-casing, removing punctuation and applying word tokenisation to the text thereafter. Some examples of the processed data is shown below.
| Dialogue Input | Dialogue Output |
| -------------- | --------------- |
| gosh if only we could find kat a boyfriend | let me see what i can do |
| take it or leave it this isn t a negotiation | fifty and you ve got your man |
| are you ready to put him in | not yet |

We remark that in this assignment, only single turn conversation is considered. Hence, a multi-turn conversation which involves a few rounds of dialogue between the parties are not considered.

### Generating the Vocabulary
To generate the vocabulary, we replace certain symbols like `\\u`, `\\i`, as well as newlines `\n` and tabs `\t`. The regular expression `re.sub(r"[^\w\s]", " ", tmp_qns)` was used to remove punctuations from the text.
```
import re
from collections import Counter

w_counter = Counter()
tmp_tuple = []
for conv in convs:
    for i in range(len(conv)-1):
        tmp_qns = id2line[conv[i]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        tmp_qns = re.sub(r"[^\w\s]", " ", tmp_qns)
        tmp_qns = [x for x in tmp_qns.split(" ") if x != ""]

        tmp_ans = id2line[conv[i+1]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        tmp_ans = re.sub(r"[^\w\s]", " ", tmp_ans)
        tmp_ans = [x for x in tmp_ans.split(" ") if x != ""]

        if len(tmp_qns) == 0 or len(tmp_ans) == 0:
            continue
        elif len(tmp_qns) <= q_len and len(tmp_ans) <= a_len:
            w_counter.update(tmp_qns)
            w_counter.update(tmp_ans)
            tmp_tuple.append((" ".join(tmp_qns), " ".join(tmp_ans)))

vocab_size = 8000
vocab_list = sorted([x for x, y in w_counter.most_common(vocab_size)])
vocab_list = ["SOS", "EOS", "PAD", "UNK"] + vocab_list

idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])
```
One peculiarity that might be observed in our pre-processing is that we used the same vocabulary for both the input and the output. This is because the dataset includes multi-turn conversations, where the output of the previous conversation could become the input of the next conversation. Hence, the vocabulary of the input and output responses have a high overlap, and we experiment training our Transformer Network using a single joint vocabulary to observe whether it is able to generalise its responses better. However, it is worth noting that while the vocabulary is shared between the input and the output responses, the encoder and decoder side do not share the same embeddings.

### Our Transformer Model
Our Transformer model makes some modifications to the original model, shown in Fig. 1, in that it trains a positional embedding layer at each layer of the encoder and decoder. In addition, it also adds a residual connection between the input embeddings at both the encoder and decoder. Apart from that, there were no further modifications made. The model uses 6 layers for both the encoder and decoder, an embedding dimension of 512, a hidden size of 512 and a feed-forward size of 2048. The sequence length at both the encoder and decoder was set to 10, which led to a total of approximately 94000 dialogue sequences, and a vocabulary of the most common 8000 words was used. A gradient clipping value of 1.0 was set during training as well. 



Before sending the data into the Transformer model, the dialogue sequences need to be converted into their corresponding integer labels. This is done via
```
tmp_i_tok = data_tuple[tmp_index][0].split(" ")
tmp_o_tok = data_tuple[tmp_index][1].split(" ")

tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
tmp_o_idx = [word2idx.get(x, UNK_token) for x in tmp_o_tok]
```
where `word2idx` is a dictionary mapping the word tokens into their corresponding integer labels.

The model has approximately 57 million parameters as returned by `seq2seq_model.summary()` function. As our model is encapsulated in a custom class and applied without using the standard `tf.keras` functionalities, the summary of the model is unable to breakdown the number of parameters at each step.
```
Layer (type)                 Output Shape              Param #
=================================================================
Total params: 57,198,080
Trainable params: 57,198,080
Non-trainable params: 0
_________________________________________________________________
```
Due to limitations on the GPU card, we accumulate the gradients manually across sub-batches of 32, then average it to apply the overall weight update across a larger batch, since we observe that larger batch sizes tend to stabilise the training of Transformer networks. 
```
with tf.GradientTape() as grad_tape:
    output_logits = model(tmp_encode, tmp_decode)
    
    tmp_losses = tf.reduce_sum(tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tmp_output, logits=output_logits), axis=1))
    
    # Accumulate the gradients. #
    tot_losses += tmp_losses
    tmp_gradients = \
        grad_tape.gradient(tmp_losses, model_params)
    acc_gradients = [
        (acc_grad+grad) for \
        acc_grad, grad in zip(acc_gradients, tmp_gradients)]
```
Following the [T5 paper](https://arxiv.org/abs/1910.10683), 2000 warmup steps with a constant learning rate was applied `step_val = float(max(n_iter+1, warmup_steps))**(-0.5)`.

### Training the Dialogue Transformer Network
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
