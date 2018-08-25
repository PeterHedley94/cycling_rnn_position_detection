#THIS FILE CONTAINS THE CALLBACKS USED DURING NN MODEL TRAINING
#WRITING TO A JSON and SENDING POSTS TO SLACK
from keras.callbacks import LambdaCallback
import json
import requests
from src.common import *
from math import ceil



def send_slack_message(message):
    if SEND_TO_SLACK:
        slack_data = {'text': str(message)}

        response = requests.post(
            webhook_url, data=json.dumps(slack_data),
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )

class logs(object):
    def __init__(self):
        # Print the batch number at the beginning of every batch.
        self.batch_print_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: print(batch))

        # Stream the epoch loss to a file in JSON format. The file content
        # is not well-formed JSON but rather has a JSON object per line.

        self.counter = 0

        self.json_log = open(JSON_LOG_FILE, mode='wt', buffering=1)
        self.json_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: self.json_log.write(
                json.dumps({'epoch': epoch, 'loss': logs['loss']
                               ,'val_loss' : logs['val_loss']}) + '\n'),
            on_train_end=lambda logs: self.json_log.close()
        )

        self.slack_callback = LambdaCallback(
            on_train_begin=lambda logs: self.start_of_training(logs),
            on_train_end=lambda logs: self.end_of_training(logs),
            on_epoch_end=lambda epoch,logs: self.update_counter(epoch,logs)
        )


    def start_of_training(self,logs):
        self.json_log = open(JSON_LOG_FILE, mode='wt', buffering=1)
        if SEND_TO_SLACK:
            slack_data = {'text': "The wolf has started to prowl"}
            self.counter = 0
            response = requests.post(
                webhook_url, data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code != 200:
                raise ValueError(
                    'Request to slack returned an error %s, the response is:\n%s'
                    % (response.status_code, response.text)
                )

    def update_counter(self,epoch,logs):
        self.counter += 1
        values_to_add = 'Epoch: ' + str(epoch+1) + '\nLoss: ' + str(logs['loss']) + '\nVal_loss: ' + str(logs['val_loss'])
        percentage = str(self.counter/NUMBER_EPOCHS * 100)
        title_string = percentage + str('% [')
        for i in range(round(self.counter/NUMBER_EPOCHS * 10)):
            title_string = title_string + '=='
        title_string = title_string + '>'
        for i in range(10-round(self.counter/NUMBER_EPOCHS * 10)):
            title_string = title_string + '__'
        title_string = title_string + ']'

        slack_data = {
            "fallback": " ",
            "fields": [
                {
                    "title": title_string,
                    "value": {'epoch': values_to_add}
                }
            ]
        }



        print(self.counter)
        print("The percentage is " + percentage)
        print("The modulus is " + str(self.counter % ceil(NUMBER_EPOCHS/20)))

        if SEND_TO_SLACK:
            if self.counter % ceil(NUMBER_EPOCHS/20) == 0:
                response = requests.post(
                    webhook_url, data=json.dumps(slack_data),
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code != 200:
                    raise ValueError(
                        'Request to slack returned an error %s, the response is:\n%s'
                        % (response.status_code, response.text)
                    )





    def end_of_training(self, logs):
        if SEND_TO_SLACK:
            slack_data = {'text': "The wolf has ended its hunt"}

            response = requests.post(
                webhook_url, data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code != 200:
                raise ValueError(
                    'Request to slack returned an error %s, the response is:\n%s'
                    % (response.status_code, response.text)
                )

"""
    def start_training_notification(self):
        self.batch_print_callback = LambdaCallback(
            on_train_begin=lambda batch, logs: print(slack_data))



    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if not np.isnan(loss) or not np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True

        # Terminate some processes after having finished model training.
        processes = ...
        cleanup_callback = LambdaCallback(
            on_train_end=lambda logs: [
                p.terminate() for p in processes if p.is_alive()])"""
