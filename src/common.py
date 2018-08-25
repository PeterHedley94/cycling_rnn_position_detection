import os
IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 200,200,3
#IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 480,640,3
sequence_length_ = 5
sequence_length = 5
pose_data_dims = (4,3)
number_outputs = 3
batch_size = 4
epochs = 100
images = False
image_height = 480#180
image_width= 640#240
no_image_channels=3
camera_model_location = os.path.join("utils",'test_camera_model.json')
train_directory = "/vol/bitbucket/ph817/imp1"#"/home/peter/Tests"
validation_directory = "/vol/bitbucket/ph817/imp3"


webhook_url = 'https://hooks.slack.com/services/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'
SEND_TO_SLACK = True
