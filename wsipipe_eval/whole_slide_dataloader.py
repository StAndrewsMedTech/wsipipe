from torch.utils.data import Dataset, DataLoader

from wsipipe.preprocess.patching import PatchSet
from wsipipe_eval.dataloader_utils import on_init, get_single_patch


class SlideDatasetPytorch(Dataset):
    def __init__(self, ps: PatchSet, augments = None) -> None:
        super().__init__()
        self.slide, self.patch_df = on_init(ps)
        self.augments = augments

    def open_slide(self):
        self.slide.open()

    def close_slide(self):
        self.slide.close()

    def __len__(self):
        return self.patch_df.shape[0]

    def __getitem__(self, idx):
        image, label = get_single_patch(idx, self.patch_df, self.slide)
        if self.augments is not None:
            image = self.augments(image)
        return image, label

def get_pytorch_slide_data_loader(ps: PatchSet, batch_size: int, augments):
    dataset = SlideDatasetPytorch(ps, augments)
    dataset.open_slide()
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return data_loader