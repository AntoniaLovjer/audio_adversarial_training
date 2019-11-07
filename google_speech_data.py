# Need directory to be top level
%cd ..
base_dir = "./"
sound_names = []
sound_file_paths = []
i = 1
for sample in trainset:
  if i%1 == 0:
    class_id, user_id, filepath = sample
    file_locations,file_names = os.path.split(filepath)
    _,sound_label = os.path.split(file_locations)
    if sound_label in ['yes','no','on','off','left','right','stop','go','up','down']:
      sound_file_paths.append(filepath)
      sound_names.append(sound_label)
  i+=1

print(sound_names[-10:])
sound_file_paths[-10:]