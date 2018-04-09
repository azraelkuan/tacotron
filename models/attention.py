"""
add attention for global style token
"""
import tensorflow as tf
from util.ops import shape_list


class MultiHeadAttention(object):

    def __init__(self,
                 num_heads,
                 num_units,
                 attention_type):
        self.num_heads = num_heads
        self.num_units = num_units
        self.attention_type = attention_type

        assert self.num_units % self.num_heads == 0

    def multi_head_attention(self, query, value):
        q, k, v = self._compute_qkv(query, value)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        outputs = self._concat_heads(outputs)
        return outputs

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads):
            t_shape = shape_list(tensor)
            dim = t_shape[-1]
            assert dim % num_heads == 0
            tensor = tf.reshape(tensor, t_shape[:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads)
        ks = split_last_dimension_then_transpose(k, self.num_heads)
        vs = tf.tile(tf.expand_dims(v, axis=1), [1, self.num_heads, 1, 1])

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        qk = tf.matmul(qs, ks, transpose_b=True)
        scale_factor = (self.num_units // self.num_heads) ** -0.5
        qk *= scale_factor
        weights = tf.nn.softmax(qk)
        context = tf.matmul(weights, vs)
        return context

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimension(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = shape_list(tensor)
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, t_shape[:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimension(outputs)

    def _compute_qkv(self, query, value):
        q = tf.layers.conv1d(query, self.num_units, 1)
        k = tf.layers.conv1d(value, self.num_units, 1)
        v = value
        return q, k, v





