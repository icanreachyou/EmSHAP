from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


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
            if self._epoch <= self._num_epoch/4:
                q = int(d*self._rmin)
            elif (self._epoch > self._num_epoch/4) and (self._epoch < self._num_epoch/4*3):
                q = int((self._epoch-self._num_epoch/4)/(self._num_epoch/4*3)*(self._rmax-self._rmin)+self._rmin)
            elif self._epoch >= self._num_epoch/4*3:
                q = int(d*self._rmax)

            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        return np.vstack(result)


def get_add_mask_fn(mask_generator):
    def fn(x):
        [mask] = tf.py_function(mask_generator, [tf.shape(x)], [x.dtype])
        # This reshape ensures that mask shape is not unknown.
        mask = tf.reshape(mask, tf.shape(x))
        return x, mask

    return fn


def get_add_marginal_masks_fn(marginal_dims):
    def fn(x):
        missing = tf.greater_equal(tf.range(tf.shape(x)[-1]), marginal_dims)
        missing = tf.cast(missing, x.dtype)
        return x, tf.zeros_like(x), missing

    return fn
