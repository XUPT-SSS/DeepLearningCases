import argparse

from sklearn.model_selection import ParameterGrid

"""
Example: shows how to organize the hyper-parameters to be grid searched
"""

params = {}
params["n_layer"] = [1, 2, 3]
params["batchsize"] = [127, 128, 200, 255, 256]
params["n_input"] = [128, 512]
params["n_units"] = [128, 512]
params["seq_length"] = [5, 25, 50]
params["random_length"] = [0, 1]
params["cudnn"] = [1, 0]
params["dropout"] = [0.0, 0.5]
params["datasize"] = [10000]
params["n_epoch"] = [10]
params["gpu"] = [0]


def arguments_organizer():
    patterns = ParameterGrid(params)
    for p in patterns:
        args = " ".join(["--" + k + "=" + str(v) for k, v in p.items()])
        # record the parameter settings in the file name
        savename = "_".join([k + "-" + str(v) for k, v in p.items()])
        # call the script to be executed with the args
        command = "python run.py " + args + " > ./log/log_" + savename + ".txt"     # modify as needed
        # print args
        print(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=20, help='learning minibatch size')
    parser.add_argument('--n_input', dest='n_input', type=int, default=100, help='n_input')
    parser.add_argument('--n_units', dest='n_units', type=int, default=200, help='n_units')
    parser.add_argument('--n_vocab', dest='n_vocab', type=int, default=10000, help='n_vocab')
    parser.add_argument('--n_layer', dest='n_layer', type=int, default=1, help='n_layer')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--seq_length', type=int, dest='seq_length', default=5, help='seq_length')
    parser.add_argument('--random_length', dest='random_length', type=int, default=0, help='random_length')
    parser.add_argument('--datasize', type=int, dest='datasize', default=10000, help='datasize')
    parser.add_argument('--cudnn', default=1, type=int, help='cudnn')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=50, help='n_epoch')
    parser.add_argument('--save_model', dest='save_model', type=int, default=0, help='n_epoch')

    args = parser.parse_args()
    print(args)
