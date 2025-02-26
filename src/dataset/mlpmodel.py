import time

import numpy as np
from singa import model, opt, device
from singa import  tensor
from singa import layer

from dataset.stream_dataloader import StreamDataloader

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class MLP(model.Model):
    def __init__(self, in_features=10, perceptron_size=16, num_classes=2):
        super(MLP, self).__init__()
        self.dimension = 2
        self.in_features = in_features
        self.perceptron_size = perceptron_size
        self.num_classes = num_classes
        self.relu1 = layer.Sigmoid()
        self.relu2 = layer.Sigmoid()
        self.relu3 = layer.Sigmoid()
        self.softmax = layer.SoftMax()
        self.linear1 = layer.Linear(self.in_features, self.perceptron_size, bias=True)
        self.linear2 = layer.Linear(self.perceptron_size, 2 * self.perceptron_size, bias=True)
        self.linear3 = layer.Linear(2 * self.perceptron_size, self.perceptron_size, bias=True)
        self.linear4 = layer.Linear(self.perceptron_size, self.num_classes, bias=True)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        y = self.relu3(y)
        y = self.linear4(y)
        y = self.softmax(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


if __name__ == '__main__':

    sgd = opt.SGD(lr=0.01, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype["float32"])
    model = MLP(in_features=10, perceptron_size=128, num_classes=2)
    model.set_optimizer(sgd)

    dev = device.get_default_device()
    dev.SetRandSeed(0)
    np.random.seed(0)


    url = "http://192.168.56.20:8094"
    table = "frappe_train"
    namespace = "train"
    dataloader = StreamDataloader(url, table, namespace)

    for epoch in range(30):
        start_time = time.time()

        # Training phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)
        model.train()
        total = 0
        for idx, batch in enumerate(dataloader, start=1):
            x = batch['id']
            y = batch['y']
            tx = tensor.Tensor(x.shape, dev, singa_dtype['float32'])
            ty = tensor.Tensor((y.shape[0],), dev, tensor.int32)

            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out, loss = model(tx, ty, dist_option = 'plain', spars = None)
            train_loss += tensor.to_numpy(loss)[0]
            train_correct += accuracy(tensor.to_numpy(out), y)
            total += y.shape[0]
        # print(model.get_params())
        print("epoch[%d]: loss:[%.6f], acc:[%d/%d = %.2f %%]" % (epoch, train_loss, train_correct, total, 100.0 * train_correct/total))






