import mxnet as mx
from mxnet import nd, gluon, init, autograd
import time
from Coil20Dataset import Coil20
from Model import Model


model = Model()

#Initialize train/val datasets
train = Coil20(data_path='train')
val = Coil20(data_path='val')

batch_size = 12

#DataLoaders
train_data = gluon.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=0)
valid_data = gluon.data.DataLoader(
    val, batch_size=batch_size, shuffle=True, num_workers=0)

#Prepare metrics
acc = mx.metric.Accuracy()
val_acc = mx.metric.Accuracy()

model.initialize(ctx=mx.cpu())

optimizer = mx.optimizer.SGD(learning_rate=0.001, momentum=0.9)
opt = gluon.Trainer(model.collect_params(), optimizer)

criterion =  gluon.loss.SoftmaxCrossEntropyLoss()
for epoch in range(10):
    train_loss = 0.0
    tic = time.time()
    acc.reset()
    val_acc.reset()
    print('Epoch : ',epoch)
    for iter, (data, label) in enumerate(train_data):
        # forward + backward
        with autograd.record():
            output = model(data)

            loss = criterion(output, label)

        loss.backward()
        # update parameters
        opt.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        acc.update(label, output)
        print("Iter %d: loss %.3f, train acc %.3f" % (
            iter, loss.mean().asscalar(), float(acc.get()[1])))
    print('Validation')
    # calculate validation accuracy
    for data, label in valid_data:
        output = model(data)
        val_acc.update(label, output)
        print("Iter %d: loss %.3f, val acc %.3f" % (
            iter, loss.mean().asscalar(), float(val_acc.get()[1])))
    print("Epoch %d: loss %.3f, train acc %.3f, val acc %.3f, in %.1f sec" % (
        epoch, train_loss / len(train_data), acc.get()[1] ,
        val_acc.get()[1], time.time() - tic))