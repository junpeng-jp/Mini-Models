import torch
from collections.abc import MutableMapping

class Trainer(object):
    def __init__(self, nEpoch, logInterval):
        self.nEpoch = nEpoch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logInterval = logInterval
        self.history = {}
        self.trainloader = None
        self.devloader = None
        self.testloader = None
        self.lossFn = None
        self.confMatrix = None

    def addDataloader(self, dataloader, loaderType):
        if loaderType not in ['train', 'dev', 'test']:
            raise AttributeError('Loader type must be either train, dev or test.')
        
        setattr(self, loaderType + "loader", dataloader)

    def addLossFn(self, lossFn):
        self.lossFn = lossFn

    def train(self, model, optimizer):
        assert self.trainloader is not None
        assert self.lossFn is not None

        model.to(self.device)
        for epoch in range(self.nEpoch):
            for i, (x, y) in enumerate(self.trainloader):
                # Forward pass
                pred = model(x.to(self.device))
                loss = self.lossFn(pred, y.to(self.device))
                self.recordHistory(epoch + 1, i + 1, loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Parameter update
                optimizer.step()    

                if (i+1) % self.logInterval == 0 or i + 1 == len(self.trainloader):
                    self._progressLog(epoch + 1, i + 1)

    def _progressLog(self, epoch, batch):
        print('Epoch [{:^3}/{:^3}]  Batch [{:^5}/{:^5}]  Loss: {:<.4f}'.format(
            epoch, self.nEpoch, 
            batch, len(self.trainloader), self.history[epoch][batch]
        ))

    def test(self, model, nClass):
        model.to(self.device)
        model.eval()
        self.confMatrix = torch.zeros((nClass, nClass), dtype=torch.int64).cpu()
        
        predictions = torch.Tensor().cpu().type(torch.int64)
        labels = torch.Tensor().cpu().type(torch.int64)

        with torch.no_grad():
            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.cpu().type(torch.int64)
                pred = model(x).cpu().type(torch.int64)
                _, pred = torch.max(pred, dim=1)
                predictions = torch.cat([predictions, pred])
                labels = torch.cat([labels, y])

        for i in range(self.confMatrix.size(0)):
            for j in range(i, self.confMatrix.size(1)):
                self.confMatrix[i, j] += torch.logical_and(predictions.eq(i), labels.eq(j)).sum().cpu()
                if i != j:
                    self.confMatrix[j, i] += torch.logical_and(predictions.eq(j), labels.eq(i)).sum().cpu()

    def recordHistory(self, epoch, batch, loss):
        if epoch not in self.history:
            self.history[epoch] = {}

        self.history[epoch][batch] = loss

if __name__ == '__main__':
    history = Trainer(3, 50)

    try:
        history.trainloader = torch.utils.data.DataLoader([1,2,3])
    except AttributeError as e:
        print(e)

    try:
        history.testloader
    except AttributeError as e:
        print(e)

    try:
        print(history.trainloader)
    except AttributeError as e:
        print(e)
    
    history.recordHistory(1, 1, 2.5)
    history.recordHistory(1, 2, 2.2)
    history.recordHistory(2, 1, 1.9)

    print(history.history)
    print(history.history[1][1])
    print(history.history[2][1])