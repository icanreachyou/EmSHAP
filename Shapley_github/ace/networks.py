from logging import debug
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers as tfl
from keras.layers.rnn import GRU
import numpy as np
import pdb
# def GRU_masked_scheme(_observed_mask, x_o, x_u):
#     _observed_mask = _observed_mask.numpy()
#     _query = 1 - _observed_mask
#
#     _query = tf.cast(_query, x_o.dtype)
#
#     return tf.multiply(_query, x_u)
#
# def _proposal_network_GRUcoupling(
#     num_features: int,
#     GRU_units: int = 100,
#     context_units: int = 64,
#     mixture_components: int = 10,
#     residual_blocks: int = 4,
#     hidden_units: int = 512,
#     activation: str = "relu",
#     dropout: float = 0.0,
#     **kwargs
# ):
# ##############################################################################################
#     def create_proposal_dist(t):
#         logits = t[..., :mixture_components]
#         means = t[..., mixture_components:-mixture_components]
#         scales = tf.nn.softplus(t[..., -mixture_components:]) + 1e-3
#         components_dist = tfp.distributions.Normal(
#             loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
#         )
#         return tfp.distributions.MixtureSameFamily(
#             mixture_distribution=tfp.distributions.Categorical(
#                 logits=tf.cast(logits, tf.float32)
#             ),
#             components_distribution=components_dist,
#         )
#
#     x_o = tfl.Input((num_features,), name="x_o")
#     observed_mask = tfl.Input((num_features,), name="observed_mask")
#     # x_u = tfl.Input((num_features,), name='x_u')
#     # query = tfl.Input((num_features,), name='query')
#
#     x_o_u = x_o
#     phi = tfl.Concatenate()([x_o, observed_mask])
#     proposal_dist = 1
#     context = 1
#     h = 0
#     # for ind in range(num_features):
#     for ind in range(1):
#         # temp = tf.add(x_o_u[:, ind], x_u[:, ind])
#         # # print(tfl.Reshape([None, 1])(temp).shape)
#         # if ind > 0 and ind<num_features-1:
#         #     x_o_u = tf.concat()([x_o_u[:, :ind], tfl.Reshape([None, 1])(temp), x_o_u[:, ind+1:]], axis=0)
#         # elif ind == 0:
#         #     x_o_u = tf.concat()([tfl.Reshape([None, 1])(temp), x_o_u[:, ind + 1:]], axis=0)
#         # elif ind == num_features-1:
#         #     x_o_u = tf.concat()([x_o_u[:, :ind], tfl.Reshape([None, 1])(temp)], axis=0)
#
#         temp1 = tf.Variable(x_o_u)
#         temp1[:, ind].assign(x_o[:, ind])
#         x_o_u = tf.convert_to_tensor(temp1)
#
#         _gru_input = tf.expand_dims(tfl.Concatenate()([phi, x_o_u]), axis=1)
#         c, h = tfl.rnn.GRU(GRU_units, return_sequences=True, return_state=True, unroll=True)(_gru_input)
#         c = tfl.Activation(activation)(c)
#         c = tfl.Dense(hidden_units)(c)
#         c = tfl.Activation(activation)(c)
#         c = tfl.Dropout(dropout)(c)
#         c = tfl.Dense(num_features * (3 * mixture_components + context_units))(c)
#         c = tfl.Reshape([num_features, 3 * mixture_components + context_units])(c)
#         # print(num_features * (3 * mixture_components + context_units))
#         # context = h[..., :context_units]
#         params = c[..., context_units:]
#         context = c[..., :context_units]
#         proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)
#
#     return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)

def gru_network(
        num_features: int,
        GRU_units: int = 64,
        num_sample: int = 1,
        context_units: int = 64,
        mixture_components: int = 10,
        residual_blocks: int = 4,
        hidden_units: int = 512,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs
):
    x = tfl.Input((num_features,), name="x_o")
    observed_mask = tfl.Input((num_features,), name="observed_mask")

    def create_proposal_dist(t):
        logits = t[..., :mixture_components]
        means = t[..., mixture_components:-mixture_components]
        scales = tf.nn.softplus(t[..., -mixture_components:]) + 1e-3
        components_dist = tfp.distributions.Normal(
            loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
        )
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=tf.cast(logits, tf.float32)
            ),
            components_distribution=components_dist,
        )

    inds = tf.where(1.0 - observed_mask == 1)[:, 0]
    inds = tf.random.shuffle(inds)
    inds = tf.cast(inds, tf.int32)

    x = x * observed_mask
    x_o = x
    # cur_x_o = tf.tile(x, [num_sample, 1])
    # cur_observed_mask = tf.tile(observed_mask, [num_sample, 1])

    phi = tfl.Concatenate()([x, observed_mask])
    _gru_input = tf.expand_dims(tfl.Concatenate()([phi, x_o]), axis=1)
    c, h = tfl.rnn.GRU(GRU_units, return_sequences=True, return_state=True, unroll=False, activation='tanh',
                       recurrent_activation='sigmoid', reset_after=True)(_gru_input)
    c = tfl.Activation(activation)(c)
    c = tfl.Dense(hidden_units)(c)
    c = tfl.Activation(activation)(c)
    c = tfl.Dropout(dropout)(c)
    c = tfl.Dense(num_features * (3 * mixture_components + context_units))(c)
    c = tfl.Reshape([num_features, 3 * mixture_components + context_units])(c)
    params = c[..., context_units:]
    context = c[..., :context_units]
    proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)

    return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)

    # for j in inds:
    #     _gru_input = tf.expand_dims(tfl.Concatenate()([phi, cur_x_o]), axis=1)
    #     c, h = tfl.rnn.GRU(GRU_units, return_sequences=True, return_state=True, unroll=True)(_gru_input)
    #     c = tfl.Activation(activation)(c)
    #     c = tfl.Dense(hidden_units)(c)
    #     c = tfl.Activation(activation)(c)
    #     c = tfl.Dropout(dropout)(c)
    #     c = tfl.Dense(num_features * (3 * mixture_components + context_units))(c)
    #     c = tfl.Reshape([num_features, 3 * mixture_components + context_units])(c)
    #     params = c[..., context_units:]
    #     context = c[..., :context_units]
    #
    #     update_inds = tf.stack([tf.range(num_sample), tf.repeat(j, num_sample)], axis=1)
    #     cur_observed_mask = tf.tensor_scatter_nd_update(cur_observed_mask, update_inds, tf.ones_like())
    #     cur_x_o = x * cur_observed_mask
    #     proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)
    #
    # return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)



def proposal_network(
    num_features: int,
    context_units: int = 64,
    mixture_components: int = 10,
    residual_blocks: int = 4,
    hidden_units: int = 512,
    activation: str = "relu",
    dropout: float = 0.0,
    **kwargs
):
    x_o = tfl.Input((num_features,), name="x_o")
    observed_mask = tfl.Input((num_features,), name="observed_mask")

    h = tfl.Concatenate()([x_o, observed_mask])
    h = tfl.Dense(hidden_units)(h)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    h = tfl.Dense(num_features * (3 * mixture_components + context_units))(h)
    h = tfl.Reshape([num_features, 3 * mixture_components + context_units])(h)

    context = h[..., :context_units]
    params = h[..., context_units:]

    def create_proposal_dist(t):
        logits = t[..., :mixture_components]
        means = t[..., mixture_components:-mixture_components]
        scales = tf.nn.softplus(t[..., -mixture_components:]) + 1e-3
        components_dist = tfp.distributions.Normal(
            loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
        )
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=tf.cast(logits, tf.float32)
            ),
            components_distribution=components_dist,
        )

    proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)
    print(proposal_dist.shape, context.shape)

    return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)


def energy_network(
    num_features: int,
    context_units: int,
    residual_blocks: int = 4,
    hidden_units: int = 128,
    activation: str = "relu",
    dropout: float = 0.0,
    energy_clip: float = 30.0,
    **kwargs
):
    x_u_i = tfl.Input((), name="x_u_i")
    u_i = tfl.Input((), name="u_i", dtype=tf.int32)
    context = tfl.Input((context_units,), name="context")

    u_i_one_hot = tf.one_hot(u_i, num_features)

    h = tfl.Concatenate()([tf.expand_dims(x_u_i, axis=-1), u_i_one_hot, context])
    # pdb.set_trace()
    h = tfl.Dense(hidden_units)(h)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    h = tfl.Dense(1)(h)

    energies = tf.nn.softplus(h)
    energies = tf.clip_by_value(energies, 0.0, energy_clip)
    negative_energies = -energies

    return tf.keras.Model([x_u_i, u_i, context], negative_energies, **kwargs)
