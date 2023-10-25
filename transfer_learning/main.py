from dependencies import *


def main():
    
    parent_dir = "Path/To/Parent_directory"
    train_loader = load_data(parent_dir, "training_directory/", 32, phase='src')
    test_loader = load_data(parent_dir, "testing_directory/", 32, phase='tar')
    print(f'Source data number: {len(train_loader.dataset)}')
    print(f'Target data number: {len(test_loader.dataset)}')

    trans_model = TransferModel().cuda()
    random_tensor = torch.randn(1, 3, 224, 224).cuda()
    output_model = trans_model(random_tensor)
    print(output_model)
    print(output_model.shape)

    dataloaders = {'src': train_loader, 'val': test_loader, 'tar': test_loader}
    param_group = []

    for k, v in trans_model.named_parameters():
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': 0.0001}]
        else:
            param_group += [{'params': v, 'lr': 0.0001 * 10}]
    optimizer = torch.optim.SGD(param_group, momentum=5e-4)

    finetune(trans_model, dataloaders, optimizer)


if __name__=="__main__":
    main()
