learning_rate = 0.001
training_iters = 10000
batch_size = 20
display_step = 500

# Network Parameters
n_input = bands 
n_steps = frames
n_hidden = 64
n_classes = labels.shape[1] # new , other , red

tf.reset_default_graph()
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))