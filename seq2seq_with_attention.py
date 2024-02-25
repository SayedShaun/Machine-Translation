import random
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout_rate, debugging=False):
        super(Encoder, self).__init__()
        self.debugging = debugging
        self.embed_layer = nn.Embedding(input_size, embed_size)
        self.rnn_layer = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, source):
        # Apply dropout and get embedding
        embedding = self.dropout_layer(self.embed_layer(source))
        print("Encoder Embedding Shape", embedding.shape)
        # Unpack RNN
        output, hidden = self.rnn_layer(embedding)

        if self.debugging:
            print("Encoder Embedding Shape", embedding.shape)
            print("Encoder Hidden Shape", hidden.shape)
            print("Encoder Output Shape", output.shape)
            
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, debugging=False):
        super(BahdanauAttention, self).__init__()
        self.debugging = debugging
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_output):
        # decoder hidden shape = (batch_size, hidden_size)
        # encoder output shape = (seq_len, batch_size, hidden_size)
       
        #add an extra dimension to match the query
        decoder_hidden = decoder_hidden.unsqueeze(0)
        #compute attention/alignment score
        score = self.V(F.tanh(self.W(decoder_hidden) + self.U(encoder_output)))
        #compute attention weights
        weights = F.softmax(score)
        #compute the context vector
        context = torch.sum(weights * encoder_output, dim=0)

        if self.debugging:
            print("Encoder Output Shape", encoder_output.shape)
            print("Decoder Hidden Shape", decoder_hidden.shape)
            print("Attention Weights Shape", weights.shape)
            print("Attention Context Shape", context.shape)

        return context, weights


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers, attention, dropout_rate, debugging=False):
        super(Decoder, self).__init__()
        self.debugging = debugging
        self.attention = attention
        self.embed_layer = nn.Embedding(output_size, embed_size)
        self.rnn_layer = nn.GRU(embed_size+hidden_size, hidden_size, n_layers, bidirectional=False)
        self.fc_layer = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, input, hidden, encoder_output):
        #Input shape (batch_size) so we have to add an extra dim (1, batch_size)
        input = input.unsqueeze(0)
        print("Decoder Input Shape", input.shape)
        embed = self.dropout_layer(self.embed_layer(input))
        print("Decoder Embedding Shape", embed.shape)

        context, weights = self.attention(hidden, encoder_output)
        context = context.unsqueeze(0)
        print("Decoder Context Shape", context.shape)

        rnn_input = torch.cat((embed, context), dim=2)
        print("Decoder RNN Input Shape", rnn_input.shape)
        #embed size (1, batch_size, embed_size)
        #encoder hidden shape (layer_size, batch_size, hidden_size)
        output, hidden = self.rnn_layer(rnn_input, hidden)
        print("Decoder Hidden Shape", hidden.shape)
        
        prediction = self.fc_layer(output)
        #prediction shape (1, batch_size, target_vocab_size) but need (batch_size, target_vocab_size)
        prediction = prediction.squeeze(0)
        print("Decoder Prediction Shape", prediction.shape)

        if self.debugging:
            print("Decoder Embedding Shape", embed.shape)
            print("Decoder Input Shape", input.shape)
            print("Decoder Prediction Shape", prediction.shape)
            print("Decoder Hidden Shape", hidden.shape)

        return prediction, hidden, context


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, source, target, tfr=0.5):
        batch_size = source.shape[1]
        seq_len = target.shape[0]

        encoder_output, encoder_hidden = self.encoder(source)
        start = target[0]
        outputs = torch.zeros(seq_len, batch_size, self.output_size).to(self.device)
        for t in range(1, seq_len):
            decoder_output, decoder_hidden, context = self.decoder(start, encoder_hidden, encoder_output)
            outputs[t] = decoder_output

            top_pred = decoder_output.argmax(1)
            start = (target[t] if random.random() < tfr else top_pred)
            
        return outputs
    
if __name__=="__main__":
    hidden_size = 512
    attention = BahdanauAttention(hidden_size, debugging=True)
    decoder_hidden = torch.zeros([32, hidden_size])
    encoder_output = torch.zeros([21, 32, hidden_size])
    context, weight = attention(decoder_hidden, encoder_output)
