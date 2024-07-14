from torch.utils.data import Dataset
from datasets import load_dataset


class GeneralDataset(Dataset):
    def __init__(
            self,
            inputs,
            targets,
            tokenizer,
            max_length=None,
            truncation=False
    ):
        self.inputs = ["<|startoftext|>" + inp + "<|endoftext|>" for inp in inputs]
        self.targets = targets
        self.tokenizer = tokenizer
        self.num_classes = len(set(self.targets))

        # Tokenize the input
        if max_length:
            self.tokenized_inputs = tokenizer(
                self.inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation
            )
        else:
            self.tokenized_inputs = tokenizer(
                self.inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        features = {'input_ids': self.tokenized_inputs.input_ids[idx],
                    'attention_mask': self.tokenized_inputs.attention_mask[idx],
                    'label': self.targets[idx]
                    }
        return features


def get_sst_dataset(
        tokenizer,
        max_length,
        truncation,
        split='val'
):
    dataset = load_dataset("glue", 'sst2')

    if split == 'train':
        dataset = GeneralDataset(
            dataset["train"]["sentence"],
            dataset["train"]["label"],
            tokenizer,
            max_length,
            truncation
        )
    elif split == 'val':
        dataset = GeneralDataset(
            dataset["validation"]["sentence"],
            dataset["validation"]["label"],
            tokenizer,
            max_length,
            truncation
        )

    return dataset
