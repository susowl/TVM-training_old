# CNN training using TVM compiler
This repo conains simple example how to use TVM compiler for CNN training.

Code realizes a simple LeNet-5 training with SGD solver. The whole training pipline realiseg in single TVM graph:

            batch<-----TrainDB
              |           |
        | -->CNN          |
        |     |           |
        |    loss <--labels
        |     |
        |    grads
        |     |
        |    momentum = momentum factor*momentum + (1.0-momentum_factor)*grads
        |----network_params = network_params - LR*momentum

Training process ends after 1 epoch. 

To start process simply run:
    python lenet_tvm.py

Expected output:
TVM:0.44873267 | accuracy:0.90625 | second per iteration:1.5032396518446336
TVM:0.5059332 | accuracy:0.875 | second per iteration:1.5032211160272118
TVM:0.4571639 | accuracy:0.875 | second per iteration:1.5032490317697649
TVM:0.36158675 | accuracy:0.96875 | second per iteration:1.5032412940138162
TVM:0.45022005 | accuracy:1.0 | second per iteration:1.5032485480447417
TVM:0.68353355 | accuracy:0.875 | second per iteration:1.503257112087071
TVM:0.6539994 | accuracy:0.8125 | second per iteration:1.5033492642064248
TVM:0.6672427 | accuracy:0.90625 | second per iteration:1.5034132038337598
TVM:0.46916655 | accuracy:0.9375 | second per iteration:1.5034249181532784

