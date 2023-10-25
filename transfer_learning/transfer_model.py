import torch
import torchvision

class TransferModel(torch.nn.Module):

    def __init__(self, raw_model : str = 'resnet50', trained_model : bool = True, num_classes : int = 1):
        
        super(TransferModel, self).__init__()
        
        self.raw_model = raw_model
        self.trained_model = trained_model
        self.num_classes = num_classes
        
        if self.raw_model == 'resnet50':
            
            self.model = torchvision.models.resnet50(pretrained = True)
            features = self.model.fc.in_features
            fc_layer = torch.torch.nn.Linear(features, num_classes)
            self.model.fc = fc_layer
        
        self.model.fc.weight.data.normal_(0, 0.005)
        self.model.fc.bias.data.fill_(0.1)

    def forward(self, tensor):
        return self.model(tensor)
    
    def predict(self, tensor):
        return self.forward(tensor)