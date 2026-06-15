from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import dga_model
from dga_reader import load_data, DataReader

flags = tf.flags

flags.DEFINE_string('data_dir', 'dga_data', 'data directory')
flags.DEFINE_string('load_model', None, 'checkpoint path without .meta/.index')

flags.DEFINE_integer('rnn_size', 50, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 30, 'character embedding size')
flags.DEFINE_integer('embed_dimension', 32, 'embedding dimension')
flags.DEFINE_string('kernels', str([2] * 20 + [3] * 10), 'CNN kernel widths')
flags.DEFINE_string('kernel_features', str([32] * 30), 'CNN kernel features')
flags.DEFINE_integer('rnn_layers', 2, 'number of LSTM layers')
flags.DEFINE_float('dropout', 0.0, 'dropout during evaluation')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('max_word_length', 70, 'maximum word length')
flags.DEFINE_integer('seed', 1021, 'random seed')

FLAGS = flags.FLAGS


def build_partial_saver(checkpoint_path):
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars = reader.get_variable_to_shape_map()

    restore_vars = []
    skipped_vars = []

    for v in tf.global_variables():
        name = v.name.split(":")[0]
        if name in ckpt_vars:
            restore_vars.append(v)
        else:
            skipped_vars.append(name)

    print("Checkpoint variables:", len(ckpt_vars))
    print("Graph variables:", len(tf.global_variables()))
    print("Restoring matched variables:", len(restore_vars))
    print("Skipping unmatched variables:", len(skipped_vars))

    if len(restore_vars) == 0:
        raise RuntimeError("No matching variables found in checkpoint.")

    return tf.train.Saver(var_list=restore_vars)


def main(_):
    if FLAGS.load_model is None:
        print("Please specify checkpoint, e.g.")
        print("python3 evaluate.py --load_model cv/autoencoder_epoch006_0.0000.model --data_dir dga_data")
        return

    if not os.path.exists(FLAGS.load_model + ".meta"):
        print("Checkpoint file not found:", FLAGS.load_model)
        return

    char_vocab, char_tensors, char_lens, max_word_length = load_data(
        FLAGS.data_dir,
        FLAGS.max_word_length
    )

    if 'test' in char_tensors:
        eval_split = 'test'
    elif 'valid' in char_tensors:
        eval_split = 'valid'
    else:
        eval_split = 'train'

    eval_reader = DataReader(
        char_tensors[eval_split],
        char_lens[eval_split],
        FLAGS.batch_size
    )

    print("Using split:", eval_split)
    print("Char vocab size:", char_vocab.size)
    print("Max word length:", max_word_length)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        with tf.variable_scope("Model"):
            m = dga_model.inference_graph(
                char_vocab_size=char_vocab.size,
                char_embed_size=FLAGS.char_embed_size,
                batch_size=FLAGS.batch_size,
                num_highway_layers=FLAGS.highway_layers,
                num_rnn_layers=FLAGS.rnn_layers,
                rnn_size=FLAGS.rnn_size,
                max_word_length=max_word_length,
                kernels=eval(FLAGS.kernels),
                kernel_features=eval(FLAGS.kernel_features),
                dropout=FLAGS.dropout,
                embed_dimension=FLAGS.embed_dimension
            )

            m.update(
                dga_model.decoder_graph(
                    m.embed_output,
                    char_vocab_size=char_vocab.size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    dropout=FLAGS.dropout
                )
            )

            m.update(
                dga_model.en_decoder_loss_graph(
                    m.input,
                    m.input_len_g,
                    m.decoder_output,
                    batch_size=FLAGS.batch_size,
                    max_word_length=max_word_length
                )
            )

        session.run(tf.global_variables_initializer())

        saver = build_partial_saver(FLAGS.load_model)
        saver.restore(session, FLAGS.load_model)

        print("Loaded matched variables from:", FLAGS.load_model)
        print("Start evaluation...")

        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        count = 0

        start_time = time.time()

        for x, y in eval_reader.iter():
            loss, loss1, loss2 = session.run(
                [m.en_decoder_loss, m.loss1, m.loss2],
                {
                    m.input: x,
                    m.input_len_g: y
                }
            )

            total_loss += loss
            total_loss1 += loss1
            total_loss2 += loss2
            count += 1

        elapsed = time.time() - start_time

        if count == 0:
            print("No evaluation batches found.")
            return

        avg_loss = total_loss / count
        avg_loss1 = total_loss1 / count
        avg_loss2 = total_loss2 / count

        print("Evaluation batches:", count)
        print("Average loss:", avg_loss)
        print("Average loss1:", avg_loss1)
        print("Average loss2:", avg_loss2)
        print("Perplexity:", np.exp(avg_loss))
        print("Elapsed time:", elapsed)
        print("Time per batch:", elapsed / count)


if __name__ == "__main__":
    tf.app.run()