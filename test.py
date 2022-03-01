def test_vis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DeepFashionTrainDataset()
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8, shuffle=True, num_workers=2)
    
    for data_dict in train_loader:
        P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
        B, C, H, W = map1.shape
        
        for b in range(B):
            mp1 = map1[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
            print(mp1.shape)
            color, mask = draw_pose_from_map(mp1)
            img = Image.fromarray(color)
            img.save('temp.jpg')

            exit()