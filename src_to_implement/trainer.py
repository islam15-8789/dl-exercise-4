import torch as t
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from tqdm.autonotebook import tqdm
import numpy as np

device = t.device("cuda" if t.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model,  # Model to be trained.
        crit,  # Loss function
        optim=None,  # Optimizer
        train_dl=None,  # Training data set
        val_test_dl=None,  # Validation (or test) data set
        cuda=True,  # Whether to use the GPU
        early_stopping_patience=-1,
    ):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.prev_f1_score = 0
        self.f1_score = 0
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save(
            {"state_dict": self._model.state_dict()},
            "checkpoints/checkpoint_{:03d}.ckp".format(epoch),
        )

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(
            "checkpoints/checkpoint_{:03d}.ckp".format(epoch_n),
            "cuda" if self._cuda else None,
        )
        self._model.load_state_dict(ckp["state_dict"])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(
            m,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            fn,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable lenght axes
                "output": {0: "batch_size"},
            },
        )

    def train_step(self, x: t.tensor, y: t.tensor) -> t.tensor:
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
        # This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        prediction = self._model(x).to(t.float32)
        y = y.to(t.float32)
        loss = self._crit(prediction, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        losses = []
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            x, y = x.to(device), y.to(device)
            # perform a training step
            loss = self.train_step(x=x, y=y)
            losses.append(loss)
        # calculate the average loss for the epoch and return it
        return sum(losses) / len(losses)

    def val_test_step(self, x: t.tensor, y: t.tensor) -> t.tensor:

        # predict
        # propagate through the network and calculate the loss and predictions
        predictions = self._model(x)
        loss = self._crit(predictions, y)
        # return the loss and the predictions
        return loss.item(), predictions.cpu().detach().numpy()

    def val_test(self):
        # set eval mode.
        self._model.eval()
        # Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.).
        # To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            y_predicted = []
            y_true = []
            losses = []
            # iterate through the validation set
            for x, y in self._val_test_dl:
                # transfer the batch to the gpu if given
                x, y = x.to(device), y.to(device)
                # perform a validation step
                loss, predictions = self.val_test_step(x=x, y=y)
                # save the predictions and the labels for each batch
                # calculate the average loss and average metrics of your choice.
                predictions = np.rint(predictions)
                y = y.cpu().detach().numpy()
                y_predicted.extend(predictions)
                y_true.extend(y)
                losses.append(loss)
                # You might want to calculate these metrics in designated functions
                self.prev_f1_score = self.f1_score
                self.f1_score = f1_score(y_true=y_true, y_pred=y_predicted, average="macro")
            print(
                "F1 Score: ",
                self.f1_score,
                end=" ==> ",
            )
            # print("Confusion matrix: ", multilabel_confusion_matrix(y_true=y_true, y_pred=y_predicted))

        # return the loss and print the calculated metrics
        return sum(losses) / len(losses)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        epoch_counter = 0
        training_losses = []
        validation_losses = []

        while True:
            # stop by epoch number
            if epoch_counter == epochs:
                break
            epoch_counter += 1
            print("Epoch: ", epoch_counter, end=" ==> ")
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            train_loss = self.train_epoch()
            training_losses.append(train_loss)
            validation_loss = self.val_test()
            validation_losses.append(validation_loss)
            print(
                "Trainging Loss: ",
                train_loss,
                " ==> ",
                "Validation loss: ",
                validation_loss,
                end="\n",
            )
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if self.f1_score >=0.60:
                self.save_checkpoint(epoch_counter)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if (
                epoch_counter >= self._early_stopping_patience
                and validation_losses[-2] - validation_losses[-1] == 0
            ):
                break
            # return the losses for both training and validation
        return training_losses, validation_losses