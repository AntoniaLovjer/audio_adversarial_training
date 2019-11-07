stest_sound_names = []
test_sound_file_paths = []
i = 1
for sample in valset:
  if i%1 == 0:
    class_id, user_id, filepath = sample
    file_locations,file_names = os.path.split(filepath)
    test_sound_file_paths.append(filepath)
    _,sound_label = os.path.split(file_locations)
    test_sound_names.append(sound_label)
  i+=1

unknowns = ["Unknown"] * len(test_sound_file_paths)

print(test_sound_names[-10:])
test_sound_file_paths[-10:]

test_features,_ = extract_features(base_dir, test_sound_file_paths, unknowns)

y_predicts = session.run(prediction, feed_dict={x: test_features, keep_prob: 1})
predicted_logit = stats.mode(np.argmax(y_predicts,1))[0][0]
list_labels = list(unique_labels)
predicted_label = list_labels[predicted_logit]
predicted_probability = stats.mode(np.argmax(y_predicts,1))[1][0] / len(y_predicts)

predicted_class = np.argmax(y_predicts,1)
actual_class = np.argmax(one_hot_encode(test_sound_names),1)

correct = 0
wrong = 0

for j in predicted_class:
  if predicted_class[j] == actual_class[j]:
    correct += 1
  else:
    wrong += 1

validation_accuracy = correct / (correct + wrong)
print(validation_accuracy)