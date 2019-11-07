def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
def RNN(x, weight, bias, keep_prob, num_layers=2):
    lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell(n_hidden, keep_prob) for _ in range(num_layers)])
    output, state = tf.nn.dynamic_rnn(lstm_layers, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = RNN(x, weight, bias, keep_prob)
# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))