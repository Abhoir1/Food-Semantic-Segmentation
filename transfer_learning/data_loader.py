from torchvision import transforms, datasets
import torch
import os

def load_data(parent_dir, path, batch_size, phase):

    mean_value = [0.485, 0.456, 0.406]
    std_value = [0.229, 0.224, 0.225]

    transform_dict = {
        'src': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean = mean_value,
                              std = std_value),
         ]),
         
        'tar': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean = mean_value,
                              std = std_value),
         ])}
    
    data = datasets.ImageFolder(root=os.path.join(parent_dir, path), transform=transform_dict[phase])
    loaded_data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=phase=='src', drop_last=phase=='tar', num_workers=4)
    return loaded_data