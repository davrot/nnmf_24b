import os
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorboard.backend.event_processing import event_accumulator  # type: ignore
import numpy as np


def get_data(path: str = "log_cnn"):
    acc = event_accumulator.EventAccumulator(path)
    acc.Reload()

    which_scalar = "Test Number Correct"
    te = acc.Scalars(which_scalar)

    np_temp = np.zeros((len(te), 2))

    for id in range(0, len(te)):
        np_temp[id, 0] = te[id].step
        np_temp[id, 1] = te[id].value
    print(np_temp[:, 1]/100)
    return np_temp


for path in glob.glob("log_*"):
    print(path)
    data = get_data(path)
    np.save("data_" + path + ".npy", data)
