from abc import ABC, abstractmethod
import numpy as np
import torch

class MaskGenerator(ABC):
    def __init__(
        self,
        seed=None,
        dtype=np.float32,
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape):
        pass


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape):
        return self._rng.binomial(1, self.p, size=shape)


class UniformMaskGenerator(MaskGenerator):
    def call(self, shape):
        assert len(shape) == 2, "expected shape of size (batch_dim, data_dim)"
        b, d = shape

        result = []
        for _ in range(b):
            q = self._rng.choice(d)
            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            # mask = -1*np.ones(d)
            mask[inds] = 1
            result.append(mask)

        return np.vstack(result)

class DynamicMaskGenerator(MaskGenerator):
    def __init__(self, _epoch, _num_epoch, _rmin, _rmax, **kwargs):
        super().__init__(**kwargs)
        self._epoch = _epoch
        self._rmin = _rmin
        self._rmax = _rmax
        self._num_epoch = _num_epoch

    def call(self, shape):
        b, d = shape    # (batch_dim, data_dim)
        q = int(d*self._rmin)
        result = []
        for _ in range(b):
            if self._epoch <= self._num_epoch/10:
                q = int(d*self._rmin)
            elif (self._epoch > self._num_epoch/10) and (self._epoch < self._num_epoch/10*9):
                q = int((self._epoch-self._num_epoch/10)/(self._num_epoch/10*9)*(self._rmax-self._rmin)+self._rmin)
            elif self._epoch >= self._num_epoch/10*9:
                q = int(d*self._rmax)

            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        return np.vstack(result)

