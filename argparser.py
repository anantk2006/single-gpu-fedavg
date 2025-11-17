"""
Command-line argument parsing.
"""

import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Distributed Training SGDClipGrad on CIFAR10 with Resnet26')
    parser.add_argument('--eta0', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--algorithm', type=str, default='local_clip',
                        choices=['local_clip', 'scaffold', 'episode', 'minibatch_clip', 'episode_mem', 'amplified_local_clip', 'amplified_scaffold'],
                        help='Optimization algorithm (default: local_clip).')
    parser.add_argument('--init_corrections', action="store_true", default=False,
                        help="Initialize corrections for ALL clients at the beginning of training for SCAFFOLD and EPISODE.")

    parser.add_argument('--world-size', type=int, default=8,
                        help='Number of processes in training (default: 8).')
    parser.add_argument('--communication-interval', type=int, default=8,
                        help='Number of local update steps (default: 8).')

    # Training
    parser.add_argument('--rounds', type=int, default=1000,
                        help='Total number of communication rounds (default: 1000).')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='How many images in each train step for each GPU (default: 32).')
    parser.add_argument('--num-evals', type=int, default=1,
                        help='How many times to evaluate the model during training. (default: 10).')
    parser.add_argument('--scaffold-comparison', action="store_true",
                        help='Whether to compare the scaffold gradients during training.')
    parser.add_argument('--init-model', type=str, default='../logs/init_model.pth',
                        help='Path to store/load the initial weights (default: ../logs/init_model.pth).')  
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Which dataset to run on (default: CIFAR10).')  
    parser.add_argument('--full_batch', action='store_true',
                        help='Do full-batch gradient descent (True) or SGD (False) (default: False).')
    parser.add_argument('--individual_evals', action='store_true',
                        help='Calculate eigenvalues at each gradient descent pass on local models (in addition to eval rounds) (default: False).')
    parser.add_argument('--model', type=str, default='resnet56',
                        help='Which model to use (default: resnet56).')
    parser.add_argument('--layers', type=int, default='2',
                        help='Number of CNN layers in the model (default: 3).')
    parser.add_argument('--loss', type=str, default='svm', choices=["svm", "cross_entropy"], help="Loss function to train model.")

    
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')
    
    parser.add_argument('--heterogeneity', type=float, default=0.0,
                        help='Data heterogeneity level, from 0 to 1 (default: 0.0).')
    

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')
    parser.add_argument('--log-fpath', type=str, default='temp.json',
                        help='Path to the log file (default: temp.json).')
    parser.add_argument('--reload-model', type=str, default="false",
                        help='Reload model from a .pth file.')
    parser.add_argument('--save-model', type=str, default=None,)

    parser.add_argument("--width", type=int, default=200, help="CIFAR hidden size for use in e_val experiments")

    return parser.parse_args()