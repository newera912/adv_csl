import tensorflow as tf
import numpy as np
import pickle
from collections import Counter


def masked_decode(preds, mask):
    """Softmax cross-entropy loss with masking."""
    loss = mask - preds
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = loss * mask
    return tf.reduce_mean(loss)


# aaaa = tf.ones([10, 5])
# aaaa = tf.reduce_mean(aaaa, axis=0)
# u = tf.random_uniform([5, 5], 1, 2)
# ji = tf.random_uniform([10, 5], 1, 2)
# uuuu = tf.stack([1/ji, ji], axis=2)
# u1 = tf.random_uniform([5, 5], 0, 2)
# ui = tf.minimum(u, 1)
# # a = tf.ones([10, 5]) * 3
# b = tf.random_uniform([5, 5], 2, 3)
# b1 = tf.random_uniform([5, 5], 0, 1)
# # # a = tf.reciprocal(a)
# # # b = tf.reciprocal(b)
# # dist = tf.distributions.Normal(loc=0.0, scale=1.0)
# # z = dist.quantile(u)
# # c = tf.pow((1.0 - tf.pow(u, tf.reciprocal(b))), tf.reciprocal(a))
# mask = np.around(np.abs(np.random.uniform(0, 0.6, [5, 5])))
# qq = masked_decode(u, mask)
#
# ds = tf.contrib.distributions
# p = ds.Normal(loc=u, scale=b)
# g = ds.Kumaraswamy(u, b)
# gg = ds.Kumaraswamy(u, b)
# pp = ds.Beta(u, b)
# # ff = tf.contrib.distributions.Kumaraswamy(3.0, 2.0).kl_divergence(tf.contrib.distributions.Kumaraswamy(3.0, 2.0))
# cc = tf.distributions.kl_divergence(p, p)
# dig = tf.digamma(2.0)
# beta = tf.exp(tf.lbeta(uuuu))
# beta_d = tf.reciprocal(beta)
# for i in range(1, 10+1):
#     a =1
# with tf.Session() as sess:
#     # aa, bb, uu, cc, zz = sess.run([a, b, u, c, z])
#     uu, ji, uii, kll, dig, beta, beta_d, aaa = sess.run([uuuu, ji, ui, cc, dig, beta, beta_d, aaaa])
# print(1)
a = np.random.beta(a=1, b=1, size=1)
b = np.random.binomial(n=1, p=a[0], size=10)
# aa = [2, 4]
# bb = [1, 2]
# oo = np.hstack((aa, bb))
# cc = np.mean(np.vstack((aa, bb)), axis=0)
# ddd = {}
# ddd[(1, 1)] = 1.0
# ddd[(2, 1)] = 1.0
# dk = ddd.keys()
# iiii = [1, 0, 0, -1, -1, -1]
# iiii2 = [-1, -1, -1, 0, 0, 0, 1, 1, -1]
# iiii3 = [1, 1, 1, 1,0]
# print(Counter(iiii))
# print(Counter(iiii).values())
# print(Counter(iiii2))
# print(Counter(iiii2).values())
# print(Counter(iiii3))
# print(Counter(iiii3).has_key(1))
# print(Counter(iiii3)[-1])
print(1)