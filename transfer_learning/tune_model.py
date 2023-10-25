import matplotlib.pyplot as plt
import numpy as np
import time
import torch


criterion = torch.nn.BCEWithLogitsLoss() 

def train_or_eval(model, phase):

    if not phase == 'src':
        model.eval()
    else:
        model.train()
        

def finetune(model, dataloaders, optimizer):

    best_model_accuracy = 0
    criterion = 0
    train_losses = []
    val_losses = []
    training_accuracy = []
    validation_accuracy = []

    for epc in range(100):
        criterion += 1

        phases_list = ['src', 'val', 'tar']

        for phase in phases_list:
            
            train_or_eval(model, phase)

            loss_value = 0
            correction = 0
            sampling_number = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    labels = labels.squeeze().float()
                    loss = criterion(outputs.squeeze(), labels.squeeze())

                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()

                loss_value += loss.item() * inputs.size(0)
                correction += torch.sum(preds == labels.data)
                sampling_number += labels.size(0)

            epoch_loss = loss_value / sampling_number
            epoch_accuracy = correction.double() / sampling_number

            if phase == 'src':
                
                train_losses.append(epoch_loss)
                training_accuracy.append(epoch_accuracy)

            elif phase == 'val':
                
                val_losses.append(epoch_loss)
                validation_accuracy.append(epoch_accuracy)
            
            if phase == 'val' and epoch_accuracy > best_model_accuracy:
                
                criterion = 0
                best_model_accuracy = epoch_accuracy
                torch.save(model.state_dict(), 'transfer_learning.pkl')

        if criterion >= 20:
            break

    training_accuracy = torch.tensor(training_accuracy)
    validation_accuracy = torch.tensor(validation_accuracy)
    training_accuracy = np.array(training_accuracy.cpu())
    validation_accuracy = np.array(validation_accuracy.cpu()) 

    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, label='Train')
    plt.plot(epochs, validation_accuracy, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()