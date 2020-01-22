from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf 


class ReversibleBlock(tf.keras.Model):
    """Single residual block contained in a _RevBlock. Each `_Residual` object has
    two _ResidualInner objects, corresponding to the `F` and `G` functions in the
    paper. This version takes in the F and G block directly, instead of constructing them. 

    This implementation is based on PyTorch's RevTorch - ReversibleBlock
    Args:
        f_block: The first residual block
        g_block: the second residual block
        split_along_axis: axis for splitting, defaults to 1
    """

    def __init__(self,
                f_block,
                g_block,
                split_along_axis=1):
        super(ReversibleBlock, self).__init__()

        self.axis = split_along_axis        
        self.f = f_block
        self.g = g_block

    def call(self, x, training=True, concat=True):
        """Apply residual block to inputs."""

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        f_x2 = self.f(x2, training=training)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, training=training)
        y2 = g_y1 + x2
        if not concat:  # For correct backward grads
            return y1, y2

        return tf.concat([y1, y2], axis=self.axis)

    def backward_grads_and_vars(self, y, dy, training=True):
        """Manually compute backward gradients given input and output grads."""
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)

        with tf.GradientTape(persistent=True) as tape:
            y = tf.identity(y)
            tape.watch(y)
            y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
            z1 = y1
            gz1 = self.g(z1, training=training)
            x2 = y2 - gz1
            fx2 = self.f(x2, training=training)
            x1 = z1 - fx2

            grads_combined = tape.gradient(
                gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
            dz1 = dy1 + grads_combined[0]
            dg = grads_combined[1:]
            dx1 = dz1

            grads_combined = tape.gradient(
                fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
            dx2 = dy2 + grads_combined[0]
            df = grads_combined[1:]

            del tape

        grads = df + dg
        vars_ = self.f.trainable_variables + self.g.trainable_variables

        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([dx1, dx2], axis=self.axis)

        return x, dx, grads, vars_


class ReversibleSequence(tf.keras.Model):
    """Single reversible block containing several `_Residual` blocks.
    Each `_Residual` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.

    This is based on PyTorch's RevTorch - ReversibleSequence
    """

    def __init__(self,
                blocks):
        """Initialize RevBlock.
        Args:
        n_res: number of residual blocks
        filters: list/tuple of integers for output filter sizes of each residual
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        bottleneck: use bottleneck residual if True
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
        """
        super(ReversibleSequence, self).__init__()
        self.blocks = blocks

    def call(self, h, training=True):
        """Apply reversible block to inputs."""
        for block in self.blocks:
            h = block(h, training=training)
        return h

    def backward_grads_and_vars(self, x, y, dy, training=True):
        """Apply reversible block backward to outputs."""

        grads_all = []
        vars_all = []

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            if i == 0:
                # First block usually contains downsampling that can't be reversed
                with tf.GradientTape() as tape:
                    x = tf.identity(x)
                    tape.watch(x)
                    y = block(x, training=training)

                    grads_combined = tape.gradient(
                        y, [x] + block.trainable_variables, output_gradients=dy)
                    dy = grads_combined[0]
                    grads_all += grads_combined[1:]
                    vars_all += block.trainable_variables
            else:
                y, dy, grads, vars_ = block.backward_grads_and_vars(
                    y, dy, training=training)
                grads_all += grads
                vars_all += vars_

        return dy, grads_all, vars_all

