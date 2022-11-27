from imports import *
from model import *
from montecarlo import *
from state import *
from controller import *
from visualizer import *
from buffer import *
    
class Trainer:
    @classmethod
    def train(cls, buffer, model):
        dataloader = DataLoader(buffer, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        model.train()
        model.to(args.device)
        
        for _ in range(2):
            for batch in dataloader:
                optimizer.zero_grad()
                x = batch['state']
                x = x.to(args.device)
                y = model(x)

                batch['value'] = batch['value'].to(args.device)
                batch['policy'] = batch['policy'].to(args.device)

                loss = model.alphaloss(y,batch['value'].float(),batch['policy'])    
                loss.backward()
                optimizer.step()
                scheduler.step()
        
class ELOEvaluator:        
    def evaluate(self, paths, num_matches = 200):
        selfplay_objs = []
        self.epochs = []
        controller = Controller()
        
        for path in paths:
            model = CRNet()
            model.load_checkpoint(path)
            selfplay_objs.append(Selfplay(model, Controller()))
            self.epochs.append(model.epoch)
            
        self.ratings = np.repeat(1000, len(selfplay_objs))
        for i in range(num_matches):
            index1, index2 = np.random.choice(len(selfplay_objs), 2, replace=False)
            winner = controller.run(
                mode = 0,
                submode = 'selfplay', 
                params = [selfplay_objs[index1], selfplay_objs[index2]],
                visualization=False
            )
            self.ratings[index1], self.ratings[index2] = self.ELO(index1, index2, winner)
        return self.ratings
        
    def ELO(self, index1, index2, winner):
        
        RA = self.ratings[index1]
        RB = self.ratings[index2]
        
        # expectations based on ratings
        EA = 1/(1+10**((RB - RA)/400))
        EB = 1/(1+10**((RA - RB)/400))
        # maximum change in ELO after one game
        K = 32
        # game true outcome
        SA = 1 if winner == 1 else 0
        SB = 1 - SA
        # new ratings
        RA += K*(SA - EA)
        RB += K*(SB - EB)
        
        return RA, RB
    
    def plot_elo_vs_epoch(self, save = False, filename = 'Rating_vs_Epoch.png'):
        Plotter.plot_splines(
            self.epochs, 
            self.ratings, 
            "Rating vs epoch",
            save,
            filename
        )

class Evaluator:
    def __init__(self, paths):
        self.paths = paths
        
    def random_evaluate(self, num_matches, mcts_iter):
        return self.__run(num_matches, mcts_iter, 3, 'Win Percent against a random player vs Epoch')
        
    def __run(self, num_matches, mcts_iter, mode, title, depth = None):
        data = {}
        self.__initialize_data(data, mcts_iter)
        delta = num_matches // 10
        num_matches_list = [delta]*10
        num_matches_list[-1] += num_matches - delta*10
        
        for i in num_matches_list:
            for j in self.paths:
                selfplay_obj = data[j]['obj']
                if depth == None:                    
                    wins = self.__play_matches([selfplay_obj], i, mode)
                else:
                    wins = self.__play_matches([selfplay_obj, depth], i, mode)
                data[j]['wins'] += wins
                data[j]['N'] += i
            x,y = self.get_plot_x_y(data)
            self.plot(x, [y], title, mcts_iter)
            
        return data
        
        
    def __initialize_data(self, data, mcts_iter):
        for path in self.paths:
            model = CRNet()
            model.load_checkpoint(path)
            selfplay_obj = Selfplay(model,None,mcts_iter)
            data[path] = {
                'obj':selfplay_obj, 
                'wins':0, 
                'epoch':model.epoch, 
                'N':0
            }
    
    def __play_matches(self, params, num_matches, mode):
        wins = Parallel(n_jobs=-1)(
            delayed(Controller().run)(
                mode,
                'selfplay',
                params,
                np.random.choice(['red','green']),
                visualization=False
            ) for _ in range(num_matches)
        )
        return sum(wins)
    
    def get_plot_x_y(self, data):
        D = {}
        for datadict in data.values():
            wins = round(100 * datadict['wins']/datadict['N'], 4)
            epoch = datadict['epoch']
            D[epoch] = wins
        wlist = sorted(D.items(), key=lambda x:x[0])
        
        X = np.array([i[0] for i in wlist])
        Y = np.array([i[1] for i in wlist])
        return X,Y
        
    
    def plot(self, X, Ylist, title, mcts_iter):
        
        if len(Ylist) == 1:
            legends = ["mcts iterations: " + str(mcts_iter)]
        else:
            legends = ["mcts iterations: " + str(mcts) for mcts in mcts_iter] 
        
        Plotter.plot_splines(
            X, 
            Ylist, 
            title,
            True,
            'wins_vs_epoch.png',
            legends = legends
        )   
        
        
class Plotter:
    @classmethod
    def plot_splines(cls, x, ylist, title, save = False, filename = None, legends = None):
        figure(figsize=(8, 6), dpi=100)
        col_list = ['red', 'blue','green']
        
        legends_is_None = False
        
        if legends == None:
            legends_is_None = True
            legends = [None]*len(ylist)
        
        xnew = np.linspace(x.min(), x.max(), 500) 
        
        for index in range(len(ylist)):        
            spl = make_interp_spline(x, ylist[index], k=3)
            plt.plot(xnew, spl(xnew), '--',color = col_list[index], label=legends[index])    
            plt.plot(x, ylist[index], 'o', color='black', markersize=4)
        
        plt.title(title)
        plt.grid()
        if not legends_is_None:
            plt.legend(loc="upper left")
        if(save):
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()
                
def selfplay_runs():
    u = Selfplay(copy.deepcopy(crnet))
    u.train_setup()
    return u.buffer.data

def perform_parallel_runs(num):
    result = Parallel(n_jobs=-1)(
        delayed(selfplay_runs)() for _ in range(num)
    )
    return result
                
def trainloop_parallel_selfplays(num_train_iter, savelist, start_epoch):

    Buffer.MAX = args.buffer_start
    delta = int((args.buffer_end - args.buffer_start)/args.slow_window_generations)
    
    for i in range(start_epoch, start_epoch+num_train_iter):
        list_of_buffer_data = perform_parallel_runs(8)
        
        for index in range(len(list_of_buffer_data)):
            MegaBuffer.extend(list_of_buffer_data[index])
        
        Trainer.train(MegaBuffer, crnet)   
        
        crnet.to(torch.device('cpu'))
        
        if int(num_train_iter/10)!=0:
            if i%int(num_train_iter/10) == int(num_train_iter/10) - 1:
                print('.',end="")
        if savelist is not None:
            if i+1 in savelist:
                crnet.save_checkpoint(
                    "checkpoints/%s/model%d.pt"%(args.FOLDER,i+1), 
                    i+1,
                    optimizer,
                    MegaBuffer,
                    scheduler
                )
        if Buffer.MAX < args.buffer_end:
            Buffer.MAX += delta

            
crnet = CRNet()
crnet.to(args.device);
MegaBuffer = Buffer(crnet)
optimizer = torch.optim.SGD(
    crnet.parameters(), lr = 0.02, momentum=0.9, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.02, max_lr=0.1)
crnet.load_checkpoint('checkpoints\\25\\model3000.pt', optimizer, MegaBuffer, scheduler)
crnet.to(torch.device('cpu'));
trainloop_parallel_selfplays(10000, list(range(3000,13001,100)), 3000)