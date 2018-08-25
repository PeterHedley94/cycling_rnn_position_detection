import os
IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 100,100,3
#IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 480,640,3
sequence_length_ = 4
sequence_length = 4
pose_data_dims = (4,3)
number_outputs = 3
batch_size = 4
epochs = 100
images = False
image_height = 480#180
image_width= 640#240
no_image_channels=3
camera_model_location = os.path.join("utils",'test_camera_model.json')
train_directory = "/home/peter/Documents/okvis_drl/build/blackfriars1_dataset"#"/home/peter/Tests"
validation_directory = "/home/peter/Documents/okvis_drl/build/blackfriars1_dataset"
