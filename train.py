saver = tf.train.Saver()
session=tf.InteractiveSession()
# Initializing the variables
session.run(tf.global_variables_initializer())

for itr in range(training_iters):    
    offset = (itr * batch_size) % (labels.shape[0] - batch_size)
    batch_x = X_train[offset:(offset + batch_size), :, :]
    batch_y = y_train[offset:(offset + batch_size), :]
    _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y, keep_prob: 0.95})

    if itr % display_step == 0:
        # Calculate batch accuracy
        acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        # Calculate batch loss
        loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        print("Iter " + str(itr) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1}) , 3))
saver.save(session, save_path = "./model/mfcc_audio.ckpt")