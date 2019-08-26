import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "The input file dir.")
flags.DEFINE_string("output_file", None, "The output file dir.")
flags.DEFINE_integer("batch_size", 256, "The size of a mini-batch.")

def main(_):
    print("The input file: " + FLAGS.input_file)
    print("The output file: " + FLAGS.output_file)
    print("The batch size: " + str(FLAGS.batch_size))

if __name__ == "__main__":
    tf.app.run()