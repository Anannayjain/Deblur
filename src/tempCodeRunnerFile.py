# constructing the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=50, 
            help='number of epochs to train the model for')
args = vars(parser.parse_args())