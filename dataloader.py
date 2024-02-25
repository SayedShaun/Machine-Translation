import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
import string
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class BaseDataset(Dataset):
    def __init__(self, file_path: str, column: str, add_sos_eos: bool=False):
        self.df = pd.read_csv(file_path)
        self.sentences = self.df[column]
        self.vocabs = build_vocab_from_iterator(
            self.token_genarator(self.sentences)
        )
        self.add_sos_eos = add_sos_eos
        if self.add_sos_eos == True:
            extra_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
            for token in extra_tokens:
                self.vocabs.append_token(token)
        else:
            extra_tokens = ["<PAD>", "<UNK>"]
            for token in extra_tokens:
                self.vocabs.append_token(token)

    def token_genarator(self, sentences: str):
        for text in sentences:
            clean_text = "".join(
                [word for word in text
                 if word not in string.punctuation]
            )
            tokens = word_tokenize(clean_text)
            yield tokens

    def text_to_sequences(self, sentences: str):
        sequence = [
            self.vocabs[token] if token in self.vocabs
            else self.vocabs["<UNK>"]
            for token in word_tokenize(sentences)
        ]
        if self.add_sos_eos == True:
            sequence = [self.vocabs["<SOS>"]] + sequence + [self.vocabs["<EOS>"]]

        return sequence

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        item = self.sentences[index]
        sequence = self.text_to_sequences(item)
        return torch.tensor(sequence)


class CombineDataset(Dataset):
    def __init__(self, data_path, eng_column, ban_column):
        self.eng_data = BaseDataset(
            file_path=data_path,
            column=eng_column,
            add_sos_eos=True
        )
        self.bng_data = BaseDataset(
            file_path=data_path,
            column=ban_column,
            add_sos_eos=True
        )

    @staticmethod
    def collate_fn(batch):
        # en, bn = zip(batch)
        en = [item[0] for item in batch]
        bn = [item[1] for item in batch]
        en_padded = pad_sequence(en, padding_value=0, batch_first=False)
        bn_padded = pad_sequence(bn, padding_value=0, batch_first=False)
        return en_padded, bn_padded

    def __len__(self):
        return len(self.eng_data)

    def __getitem__(self, index):
        eng_item = self.eng_data[index]
        bng_item = self.bng_data[index]
        return eng_item, bng_item


if __name__ == "__main__":
    dataset = CombineDataset(
        data_path="Data/BAN-Cap_caption_data.csv",
        eng_column="english_caption",
        ban_column="bengali_caption"
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=32,
        collate_fn=dataset.collate_fn
    )

    english, bangla = next(iter(dataloader))
    print(english.shape, bangla.shape)
    print(len(dataset.bng_data.vocabs))
