import os
IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 50,50,3
#IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS = 480,640,3
sequence_length_ = 5
sequence_length = 5
pose_data_dims = (4,3)
number_outputs = 3
batch_size = 3
BATCH_SIZE = 3
epochs = 100
NUMBER_EPOCHS = epochs
images = False
image_height = 480#180
image_width= 640#240
no_image_channels=3
camera_model_location = os.path.join("utils",'test_camera_model.json')
train_directory = "/vol/bitbucket/ph817/imp1"#"/home/peter/Tests"#"/home/peter/Documents/okvis_drl/build/imp4"
validation_directory = "/vol/bitbucket/ph817/imp3"#"/home/peter/Documents/okvis_drl/build/imp4"#


webhook_url = 'https://hooks.slack.com/services/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'
SEND_TO_SLACK = True


#For Model outputs
CHECKPOINTS_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','checkpoints')
MODEL_SAVE_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','models')
OLD_MODELS_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','old_models')
TENSORBOARD_LOGS_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','logs')
TENSORBOARD_OLD_LOGS_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','old_logs')
INTERMEDIATE_FILE = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','checkpoints','intermediate.hdf5')
JSON_LOG_FILE = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','loss_log.json')
JSON_OLD_LOGS_FOLDER = os.path.join('/vol/bitbucket/ph817','MODEL_OUTPUTS','old_json')
