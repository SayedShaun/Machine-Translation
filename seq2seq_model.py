import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout_rate, debugging=False):
        super(Encoder, self).__init__()
        self.debugging = debugging
        self.embed_layer = nn.Embedding(input_size, embed_size)
        self.rnn_layer = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, source):
        # Apply dropout and get embedding
        embedding = self.dropout_layer(self.embed_layer(source))
        # Unpack RNN
        output, hidden = self.rnn_layer(embedding)

        if self.debugging:
            print("Encoder Embedding Shape", embedding.shape)
            print("Encoder Output Shape", output.shape)
            print("Encoder Hidden Shape", hidden.shape)
            
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers, dropout_rate, debugging=False):
        super(Decoder, self).__init__()
        self.debugging = debugging
        self.embed_layer = nn.Embedding(output_size, embed_size)
        self.rnn_layer = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)
        self.fc_layer = nn.Linear(hidden_size*2, output_size)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, input, encoder_hidden):
        #Input shape (batch_size) so we have to add an extra dim (1, batch_size)
        input = input.unsqueeze(0)
        embed = self.dropout_layer(self.embed_layer(input))
        #embed size (1, batch_size, embed_size)
        #encoder hidden shape (layer_size, batch_size, hidden_size)
        output, hidden = self.rnn_layer(embed, encoder_hidden)
        prediction = self.fc_layer(output)
        #prediction shape (1, batch_size, target_vocab_size) but need (batch_size, target_vocab_size)
        prediction = prediction.squeeze(0)

        if self.debugging:
            print("Decoder Embedding Shape", embed.shape)
            print("Decoder Input Shape", input.shape)
            print("Decoder Prediction Shape", prediction.shape)
            print("Decoder Hidden Shape", hidden.shape)

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = output_size
        self.device = device

    def forward(self, source, target, tfr=0.5):
        batch_size = source.shape[1]
        seq_len = target.shape[0]

        encoder_output, encoder_hidden = self.encoder(source)
        start = target[0]
        outputs = torch.zeros(seq_len, batch_size, self.output_size).to(self.device)
        for t in range(1, seq_len):
            decoder_output, decoder_hidden = self.decoder(start, encoder_hidden)
            outputs[t] = decoder_output

            top_pred = decoder_output.argmax(1)
            start = (target[t] if torch.rand([1]) < tfr else top_pred)
            
        return outputs
