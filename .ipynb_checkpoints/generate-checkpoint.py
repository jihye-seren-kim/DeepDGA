from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import dga_model
from dga_reader import load_data

flags = tf.flags

flags.DEFINE_string('load_model', None, 'checkpoint path without .meta')
flags.DEFINE_string('data_dir', 'dga_data', 'data directory')
flags.DEFINE_integer('num_samples', 300, 'how many domains to generate')
flags.DEFINE_integer('batch_size', 64, 'must match training batch size')
flags.DEFINE_integer('random_dimension', 32, 'generator random input dimension')

flags.DEFINE_integer('rnn_size', 50, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 30, 'character embedding size')
flags.DEFINE_integer('embed_dimension', 32, 'embedding dimension')
flags.DEFINE_string('kernels', str([2] * 20 + [3] * 10), 'CNN kernel widths')
flags.DEFINE_string('kernel_features', str([32] * 30), 'CNN kernel features')
flags.DEFINE_integer('rnn_layers', 2, 'number of LSTM layers')
flags.DEFINE_float('dropout', 0.0, 'dropout during generation')
flags.DEFINE_integer('max_word_length', 70, 'maximum word length')
flags.DEFINE_integer('seed', 1021, 'random seed')

FLAGS = flags.FLAGS


def clean_domain(s):
    s = s.replace(" ", "")
    s = s.replace("\x00", "")
    return s.strip()


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
        print("python3 generate.py --load_model cv/gl_epoch004_1.9666.model --data_dir dga_data")
        return

    if not os.path.exists(FLAGS.load_model + ".meta"):
        print("Checkpoint file not found:", FLAGS.load_model)
        return

    char_vocab, char_tensors, char_lens, max_word_length = load_data(
        FLAGS.data_dir,
        FLAGS.max_word_length
    )

    print("Loaded vocabulary. Char vocab size:", char_vocab.size)
    print("Max word length:", max_word_length)

    np_random = np.random.RandomState(FLAGS.seed)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)

        with tf.variable_scope("Model"):
            gen_model = dga_model.genearator_layer(
                batch_size=FLAGS.batch_size,
                input_dimension=FLAGS.random_dimension,
                max_word_length=max_word_length,
                embed_dimension=FLAGS.embed_dimension
            )

            dec_model = dga_model.decoder_graph(
                gen_model.gl_output,
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

            gen_model.update(dec_model)

        session.run(tf.global_variables_initializer())

        saver = build_partial_saver(FLAGS.load_model)
        saver.restore(session, FLAGS.load_model)

        print("Loaded matched variables from", FLAGS.load_model)
        print("Generated domains:")

        generated_count = 0

        while generated_count < FLAGS.num_samples:
            generator_input = np_random.rand(
                FLAGS.batch_size,
                FLAGS.random_dimension
            )

            generated = session.run(
                gen_model.generated_dga,
                {
                    gen_model.gl_input: generator_input
                }
            )

            for row in generated:
                domain = clean_domain(char_vocab.change(row))

                if domain:
                    print(domain)
                    generated_count += 1

                if generated_count >= FLAGS.num_samples:
                    break


if __name__ == "__main__":
    tf.app.run()