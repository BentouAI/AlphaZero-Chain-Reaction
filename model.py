from imports import *

'''
Description of the encoded state
Size of the state: M*N*7
channel 1 : stores the MxN map where red orbs are 1 in number
channel 2 : stores the MxN map where red orbs are 2 in number
channel 3 : stores the MxN map where red orbs are 3 in number
channel 4 : stores the MxN map where green orbs are 1 in number
channel 5 : stores the MxN map where green orbs are 2 in number
channel 6 : stores the MxN map where green orbs are 3 in number
channel 7 : np.ones((M,N)) if red player's turn and np.zeros((M,N)) if it is green player's turn
'''

class SmallBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallBlock, self).__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels)
        )
    def forward(self, X):
        return self.model(X)
        
class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ResnetBlock, self).__init__()
        self.model = torch.nn.Sequential(
            SmallBlock(in_channels, mid_channels),
            torch.nn.ReLU(),
            SmallBlock(mid_channels, in_channels)
        )
    def forward(self, X):
        Y = self.model(X)
        Y = Y + X
        Y = torch.nn.ReLU()(Y)
        return Y
    
class DropoutBlock(torch.nn.Module):
    def __init__(self, in_units, out_units):
        super(DropoutBlock, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_units, out_units),
            torch.nn.BatchNorm1d(out_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=args.dropout)
        )
    def forward(self, X):
        return self.model(X)
        
class CRNet(torch.nn.Module):
    def __init__(self, H=[200,100], num_channels = 32):
        
        # input shape: batch_size x 7 x args.M x args.N
        
        super(CRNet, self).__init__()
        self.epoch = None
        
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(7, num_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU()
        )
        self.middle_blocks = torch.nn.Sequential(
            *[ResnetBlock(num_channels,num_channels) for _ in range(5)]
        )
        self.dropout_blocks = torch.nn.Sequential(
            DropoutBlock(num_channels * args.M * args.N, H[0]),
            DropoutBlock(H[0], H[1])
        )
          
        self.model = torch.nn.Sequential(
            self.initial_block,
            self.middle_blocks,
            torch.nn.Flatten(start_dim=1),
            self.dropout_blocks
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(H[1], H[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H[1], 1),
            torch.nn.Tanh()
        )
        
        self.my_policy_head = torch.nn.Sequential(
            torch.nn.Linear(H[1], H[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H[1], args.M*args.N),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, X):
        # whenever the batch size is 1, or the batch dimension doesn't exist, then it is eval mode
        # else it should be train mode
        if X.dim() == 3:
            X = X.unsqueeze(0)        
        Y = self.model(X)
        v = self.value_head(Y)
        my_p = self.my_policy_head(Y).reshape(-1,args.M,args.N) 
        return {'value':v, 'policy':my_p}
        
    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # converts state dictionary which contains 2d array and 
    # turn that info into the 7xMxN info to be fed to the NN
    def state_array_view_to_tensor(self, state_array_view):
        turn = state_array_view['player_turn']
        turn_arr = np.ones((args.M,args.N)) if turn == 'red' else np.zeros((args.M,args.N))
        L = [(state_array_view['array_view'] == i).astype('float') for i in [1,2,3,-1,-2,-3]]
        L.append(turn_arr)
        return torch.from_numpy(np.array(L)).float()
    
    def tensor_to_state_array_view(self, encoded_tensor):
        if(torch.all(encoded_tensor[-1] == torch.ones((3,3))).item()):
            turn = 'red'
        else:
            turn = 'green'
        t = encoded_tensor[:-1]
        out = {}
        out['array_view'] = np.array(t[0]+2*t[1]+3*t[2]-t[3]-2*t[4]-3*t[5])
        out['player_turn'] = turn
        return out
    
    def load_checkpoint(self, path, optimizer = None, buffer = None, scheduler = None):
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if buffer is not None:
            buffer.data = checkpoint['buffer']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def save_checkpoint(self, path, epoch, optimizer, buffer, scheduler):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'buffer':buffer.data,
            'scheduler_state_dict':scheduler.state_dict()
            }, path)
        
    def alphaloss(self, y, mcts_val, mcts_policy):
        loss1 = torch.nn.MSELoss()(
            y['value'].reshape(-1), 
            mcts_val
            )
        loss2 = torch.nn.CrossEntropyLoss()(
            y['policy'].reshape(-1,args.M*args.N),
            mcts_policy.reshape(-1,args.M*args.N)
            )
        loss = loss1 + loss2
        
        return loss