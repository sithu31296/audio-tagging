import pickle
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class ESC50(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.num_classes = 50
        self.transform = transform
        with open(root, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Found {len(self.data)} audios in {root}.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        data = self.data[index]
        audio, target = data['audio'], data['target']

        if self.transform:
            audio = self.transform(audio)
        return audio, target.long()


if __name__ == '__main__':
    dataset = ESC50('C:\\Users\\sithu\\Documents\\Datasets\\ESC50\\mel\\validation128mel1.pkl')
    dataloader = DataLoader(dataset, 4, True)
    audio, target = next(iter(dataloader))
    print(audio.shape, target)