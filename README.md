# Australian weather prediction with federated learning

This project is about rainfall prediction in Australia. It is based on the Flower framework with modifications to save the global model after each iteration and evaluate the trained global model.

## About Dataset

The dataset used in this project is sourced from Kaggle and consists of Australian rainfall data, comprising approximately 140,000 records.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/A5he1ter/AUC-Weather-Prediction.git && cd AUC-Weather-Prediction
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. We need to specify the partition id to
use different partitions of the data on different nodes. To do so simply open two more terminal windows and run the
following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1
```

Start client 3 in the third terminal:

```
python3 client.py --partition-id 2
```

If you're using a Linux or Unix-like system, you can directly execute the `run.sh` file.

## Thanks

Thank you to the authors of the Flower framework and to the providers of the Australian rainfall data.
