import tensorflow as tf

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.contrib.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

a = next_element + 3

with tf.Session() as session:
    session.run(iterator.initializer, feed_dict={max_value: 10})
    try:
        while True:
            value = session.run(a)
            print(value)
    except tf.errors.OutOfRangeError:
        pass