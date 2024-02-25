import torch
from nltk import word_tokenize

def translate_sentence(sentence, model, dataset, device):
    model.eval()
    with torch.no_grad():
        tokens = tokens = [token for token in word_tokenize(sentence)]
        print(tokens)
        tokens = ["<SOS>"] + tokens + ["<EOS>"]
        indices = dataset.eng_data.vocabs.lookup_indices(tokens)
        print(indices)
        tensor = torch.LongTensor(indices).unsqueeze(-1).to(device)
        print(tensor.shape)
        encoder_output, encoder_hidden = model.encoder(tensor)
        print(encoder_hidden.shape)
        inputs = dataset.bng_data.vocabs.lookup_indices(["<SOS>"])
        for _ in range(10):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(inputs_tensor, encoder_hidden, encoder_output)
            predicted_token = output.argmax(1)
            inputs.append(predicted_token)
            if predicted_token == dataset.bng_data.vocabs["<EOS>"]:
                break
        tokens = dataset.bng_data.vocabs.lookup_tokens(inputs)
    return " ".join(tokens[1:])

sentence = "a girl is walking"
translate_sentence(sentence)