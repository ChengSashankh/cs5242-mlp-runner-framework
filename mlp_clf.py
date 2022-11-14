import os

import torch
import json
from torch import nn

from base.dataset import DatasetReader
from base.constants import OUTPUT_DIR, LOGFILE
from log.logger import Logger
from simple_average_classifier import SimpleMLPTextClassifier


def run_model(structure, model_name, lr, epochs, input_dim, batch_size, logger, X_train, y_train, X_val, y_val, X_test, y_test):
    # Define and run the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleMLPTextClassifier(checkpoint_freq=20, loss_fn=loss_fn,
                                    model_spec=structure, name=model_name,
                                    sent_cleaner_conf=None, device=device, input_dim=input_dim, batch_size=batch_size, logger=_logger, alpha=lr)
    try:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        model.run_training(X_train, y_train, X_val, y_val, optim, epochs=epochs)

        logger.log("Next line is test loss - ignore the train loss there")
        model.cal_loss_and_accuracy(X_test, y_test, 11.1111, 0)

        return 0
    except KeyboardInterrupt:
        logger.log('Stopped - Next line is test loss - ignore the train loss there')
        model.cal_loss_and_accuracy(X_test, y_test, 11.1111, 0)
        return 1


def get_configs(config_file="config.json"):
    with open(config_file, 'r') as fp:
        return json.load(fp)["configs"]


####################################################################################
# Main                                                                             #
####################################################################################

if __name__ == "__main__":
    configs = get_configs("config.json")

    for idx_config, config in enumerate(configs):
        print(f"Starting with model {idx_config}: {config['model_name']}")
        os.makedirs(f"outputs/{config['model_name']}", exist_ok=True)
        _logger = Logger(f"{OUTPUT_DIR}/{config['model_name']}", LOGFILE, "mlp_clf.py")

        dataset_reader = DatasetReader(config['embedding_type'], _logger)
        _X_train, _X_test, _y_train, _y_test, _X_val, _y_val = dataset_reader.read(config['data_path'], config['simple'], config['model_name'])
        outcome = run_model(config['structure'], config['model_name'], config['lr'], config['epochs'], config['input_dim'], config['batch_size'], _logger, _X_train, _y_train, _X_val, _y_val, _X_test, _y_test)

        if outcome == 0:
            _logger.log ("Completed training and testing")
        else:
            _logger.log ("Failed")
