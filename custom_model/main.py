from libraries import *


def DistributeClassWeight(num_classes, c=1.02):
    
    loaded_model = extract_data("Path/To/Training_images/", "Path/To/Annotated_training_images/", batch_size = 2000)
    division_factor = len((next(loaded_model)[1]).flatten())
    return 1 / (np.log(c + np.bincount((next(loaded_model)[1]).flatten(), minlength=num_classes)/ division_factor))


def main():

    enet = ENet(12)
    max_epochs = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    enet = enet.to(device)

    weight_dist = DistributeClassWeight(103)

    CE_threshold = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_dist).to(device))
    opt_Adam = torch.optim.Adam(enet.parameters(), lr=5e-4, weight_decay=2e-4)

    train_losses = []
    eval_losses = []

    train_batch = 2000 // 50
    eval_batch = 2135 // 50
    loaded_model = extract_data("Path/To/Training_images/", "Path/To/Annotated_training_images/", 50)
    eval_model = extract_data("Path/To/Test_images/", "Path/To/Annotated_test_images/", 50)
    epch = max_epochs*10^3

    for e in range(0, epch):
        
        train_loss = 0
        
        enet.train()
        
        for _ in tqdm(range(train_batch)):
            X_batch, mask_batch = next(loaded_model)
            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            opt_Adam.zero_grad()

            out = enet(X_batch.float())
            
            loss_criterion = CE_threshold(out, mask_batch.long())
            loss_criterion.backward()
            opt_Adam.step()

            train_loss += loss_criterion.item()
            
        train_losses.append(train_loss)
        
        if e % 5 == 0:
            with torch.no_grad():
                enet.eval()
                
                eval_loss = 0

                # Validation loop
                for _ in tqdm(range(eval_batch)):
                    inputs, labels = next(eval_model)

                    
                    inputs, labels = inputs.to(device), labels.to(device)
                        
                    
                    out = enet(inputs)
                    
                    out = out.data.max(1)[1]
                    
                    eval_loss += (labels.long() - out.long()).sum()
                    
                
                print ()
                print ('Loss {:6f}'.format(eval_loss))
                
                eval_losses.append(eval_loss)
            
            checkpoint = {
                'epch' : e,
                'state_dict' : enet.state_dict()
            }
            torch.save(checkpoint, 'checkpoints.pth')

    state_dict = torch.load('checkpoints.pth')['state_dict']
    enet.load_state_dict(state_dict)


    fname = "testing_image"
    tmg_ = plt.imread("Path/To/Training_images/" + fname + '.jpg')
    tmg_ = cv2.resize(tmg_, (512, 512), cv2.INTER_NEAREST)
    tmg = torch.tensor(tmg_).unsqueeze(0).float()
    tmg = tmg.transpose(2, 3).transpose(1, 2).to(device)

    enet.to(device)
    with torch.no_grad():
        out1 = enet(tmg.float()).squeeze(0)

    smg_ = Image.open("Path/To/Annotated_training_images/" + fname + '.png')
    smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)


    b_ = out1.data.max(0)[1].cpu().numpy()  

    # Save the parameters
    checkpoint = {
        'epoch' : e,
        'state_dict' : enet.state_dict()
    }
    torch.save(checkpoint, 'checkpoints.pth')
    
    # Save the model
    torch.save(enet, 'model.pt')

if __name__=="__main__":
    main()