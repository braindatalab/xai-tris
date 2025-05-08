from typing import Tuple, Union

from numpy.typing import ArrayLike
import pandas as pd
import torch


class NAMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 X: Union[ArrayLike, pd.DataFrame],
                 y: Union[ArrayLike, pd.DataFrame],
                 w: Union[ArrayLike, pd.DataFrame] = None,
                 device: str = 'cpu'):
        """Dataset for NAMs that handles default weights.

        Args:
            X (Union[ArrayLike, pd.DataFrame]): Feature array.
            y (Union[ArrayLike, pd.DataFrame]): Target array.
            w (Union[ArrayLike, pd.DataFrame]): Weight array.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        else:
            self.X = X

        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, requires_grad=False, dtype=torch.float)
        else:
            self.y = y

        if not w:
            self.w = torch.clone(self.y)
            self.w[~torch.isnan(self.w)] = 1.0
            self.w[torch.isnan(self.w)] = 0.0
        else:
            self.w = torch.tensor(w, requires_grad=False, dtype=torch.float)

        if len(self.y.shape) > 1:
            # In multitask setting, set missing labels to 0. The loss
            # contributions from these examples will get zeroed out downstream
            # but leaving nan values will cause a crash.
            self.y[self.y != self.y] = 0.0
        else:
            # Create task dimension in single task setting for consistency.
            self.y = self.y.unsqueeze(1)
            self.w = self.w.unsqueeze(1)
        
        self.X = self.X.to(torch.device(device))
        self.y = self.y.to(torch.device(device))
        self.w = self.w.to(torch.device(device))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[ArrayLike, ...]:
        return self.X[idx], self.y[idx], self.w[idx]