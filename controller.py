from imports import *
from state import *
from visualizer import *
from model import *
from buffer import *
from montecarlo import *

# controls state transition when an event occurs
class Controller:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = State()
        self.green_orbs = 0
        self.red_orbs = 0
                 
    # given a state dictionary, find out what are the actions possible from this state
    def get_actions(self, state):
        State_view = state['array_view']
        possible = State_view>=0 if state['player_turn'] == 'red' else State_view<=0
        return [(i,j) for i in range(args.M) for j in range(args.N) if possible[i][j]]
    
    # getting the next state, given state in dictionary form and an action
    def get_next_state(self, state, action):
        return self.get_next_state_objmethod(
            State(state), 
            action
        )
    
    # getting the next state, given state object and action
    def get_next_state_objmethod(self, state_obj, action, myorbs = None, opporbs = None):
        # the following call changes the state_obj, performing the event on it
        return_val = self.event(action[0], action[1], state_obj, myorbs, opporbs)
        return state_obj, return_val
        
    # checks if an event is valid given a state
    def is_valid_event(self, state, row, col, turn_color = None):
        # assume it's a valid event
        valid = True
        if turn_color is None:
            turn_color = int(state.turn == 'red')*2 - 1
        
        # the event is invalid given these conditions
        # 1) row or column index is invalid 
        # 2) orbs of opposite color already exist at the given node
        
        if(row<0 or row>args.M-1 or col<0 or col>args.N-1):
            valid = False
        elif((state.orbs[row][col]>0) and state.color[row][col] == -turn_color):
            valid = False
                
        return valid
        
    
    def event(self, row, col, state = None, myorbs = None, opporbs = None):
        # a mouse event has occurred for given row and given col indices
        
        own_state = False
        
        if state is None:
            state = self.state
            own_state = True
            
        # checking if the event is valid
        turn_color = int(state.turn == 'red')*2 - 1
        valid = self.is_valid_event(state, row, col, turn_color)
        winner = None
        
        if not valid:
            print('invalid event')
        else:
            '''
            get the next state given the event, this block contains
            the orb explosion logic, it's bfs, we maintain a queue
            and keep on inserting neighbours if the current orb explodes
            handling one neighbour before the other is fine as long as we
            don't handle any other cell between handling them
            |A|B|C|  When E explodes here, we add D, B, F and H to the queue
            |D|E|F|  When we handle D, it can never influence B, F and H 
            |G|H|I|  directly
            '''
            
            # keep track of our orbs and opponent's orbs in order to keep track
            # of when the game is over
            if myorbs == None and opporbs == None:
                if own_state is False:
                    myorbs = np.sum(state.orbs[state.color==turn_color])
                    opporbs = np.sum(state.orbs[state.color==-turn_color])        
                else:
                    if state.turn == 'red':
                        myorbs = self.red_orbs
                        opporbs = self.green_orbs
                    else:
                        myorbs = self.green_orbs
                        opporbs = self.red_orbs
                    
            '''
            row and col are the indices for the event cell
            the '1' in the following line of code indicates
            that one orb would be added in that cell
            '''
            queue = [(row, col, 1)]
    
            # one of my orbs will increase at the (row, col) cell
            myorbs += 1
    
            while len(queue)>0:

                row,col,val = queue.pop(0)
                
                # if the current cell which is being handled has orbs of opposite color, then
                # we need to decrease them from our opporbs count, 
                # because they will change to our color
                if state.color[row][col] == -turn_color:
                    opporbs -= state.orbs[row][col]
                    myorbs += state.orbs[row][col]
                    
                '''
                update the orbs count and set the orbs color to be
                the player's color
                '''
                
                state.orbs[row][col] += val
                state.color[row][col] = turn_color
                orbs = state.orbs[row][col]
                degree = state.degree[row][col]
        
                # an explosion of some node has happened
                if orbs >= degree:
                    # quotient number of orbs would go to the neighbours
                    Quotient = orbs // degree
                    # remaining orbs will remain
                    state.orbs[row][col] = orbs % degree
                    
                    # adding the neighbours to the queue because of the explosion
                    if row!=0:
                        queue.append((row-1, col, Quotient))
                    if row!=args.M-1:
                        queue.append((row+1, col, Quotient))
                    if col!=0:
                        queue.append((row, col-1, Quotient))
                    if col!=args.N-1:
                        queue.append((row, col+1, Quotient))
                
                # the opponent is left with no orbs
                if opporbs<=0 and myorbs!=1:
                    winner = state.turn
                    break
                    
            if own_state is True:
                if state.turn == 'red':
                    self.red_orbs = myorbs
                    self.green_orbs = opporbs
                else:
                    self.green_orbs = myorbs
                    self.red_orbs = opporbs
            
            state.turn = 'green' if state.turn == 'red' else 'red'        
        
        return winner
                
    def eventloop(self):
        # the following line is a way around busy waiting in pygame (I think)
        event = pygame.event.wait()
        # events can be many, but the useful ones in our case are:
        # 1) clicking on the close button on the window, i.e. Quit event
        # 2) clicking somewhere
        if event.type == pygame.QUIT:
            return "quit"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            row = pos[1]//self.visualizer.cell_edge
            col = pos[0]//self.visualizer.cell_edge
            self.last_event_action = (row,col)
            winner = self.event(row, col)
            return winner
        else:
            return None
    
    # private function
    def __handle_one_player_mode(self, submode, params):
        self.last_event_action = None
        if submode == 'path':
            if len(params)!=1:
                raise Exception("Length of the list of paths should be one")
            net = CRNet()
            net.load_checkpoint(params[0])
            return Selfplay(net)
        elif submode == 'selfplay':
            if len(params)!=1:
                raise Exception("Length of the list of paths should be one")
            params[0].reset()
            return params[0]
        else:
            raise Exception("Provide a suitable submode")
            
    # private function
    def __handle_zero_player_mode(self, submode, params):
        if len(params)!=2:
            raise Exception("Length of the list of paths should be two")
        # either specify the path to the models or pass selfplay objects
        if submode == 'path':
            net1 = CRNet()
            net2 = CRNet()
            net1.load_checkpoint(params[0])
            net2.load_checkpoint(params[1])
            return Selfplay(net1), Selfplay(net2)
        elif submode == 'selfplay':
            return params[0], params[1]
        else:
            raise Exception("Provide a suitable submode")
            
    def __handle_random_agent(self, submode, params):
        if submode == 'path':
            net = CRNet()
            net.load_checkpoint(params[0])
            return Selfplay(net)
        elif submode == 'selfplay':
            return params[0]
        else:
            raise Exception("Provide a suitable submode")
    
    # run the game, either 2 human players, or 1 or None
    # mode :
    # 3: against random agent
    # 2: two human players
    # 1: one human player, submodes exist, either path or selfplay
    # 0: zero human players, submodes exist, either path or selfplay
    def run(self, mode, submode = None, params = None, turn = 'red', visualization = True):
    
        s1 = None
        s2 = None
        winner = None
        
        if mode == 2:
            if(visualization == False):
                raise Exception("Two human players mode requires visualization")
        elif mode == 1:
            if(visualization == False):
                raise Exception("One human player mode requires visualization")
            s1 = self.__handle_one_player_mode(submode, params)
        elif mode == 0:
            s1, s2 = self.__handle_zero_player_mode(submode, params)
        elif mode == 3:
            s1 = self.__handle_random_agent(submode, params)
        else:
            raise Exception("Provide a suitable mode")
            
        # start the state with an empty board and red plays first
        self.reset()
        if(visualization):
            self.visualizer = Visualizer()

        def handle_event_loop(winner):
            while True:
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    winner = 'quit'
                    pygame.quit()
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    break
            return winner
        
        while True:
            # handles the graphics part
            if(visualization):
                self.visualizer.gameloop(self.state.get_array_view())
            if mode == 3:
                if self.state.turn == turn:
                    actions_possible = self.get_actions(self.state.get_array_view())
                    action = random.choice(actions_possible)
                else:
                    action = s1.play_given_state(self.state.get_array_view())
                winner = self.event(action[0], action[1])
                if(visualization):
                    winner = handle_event_loop(winner)
            elif mode == 2:            
                winner =  self.eventloop()
            elif mode == 1:
                if self.state.turn == turn:
                    winner = self.eventloop()
                else:
                    if self.last_event_action is not None:
                        if s1.montecarlo_tree.root.children[self.last_event_action].state is None:
                            s1.reset(self.state.get_array_view())
                        else: 
                            s1.montecarlo_tree.update(self.last_event_action)
                    action = s1.play_own_state()
                    s1.montecarlo_tree.update(action)
                    winner = self.event(action[0], action[1])
            else:
                if self.state.turn == turn:
                    action = s1.play_given_state(self.state.get_array_view())
                else:
                    action = s2.play_given_state(self.state.get_array_view())
                winner = self.event(action[0], action[1])
                if(visualization):
                    winner = handle_event_loop(winner)
                
            if winner is not None:
                pygame.quit()
                
                if (mode == 3 or mode == 4) and winner is not 'quit':
                    return 0 if turn == winner else 1
                
                if winner is 'quit':
                    return 0
                elif winner is 'red':
                    return 1
                elif winner is 'green':
                    return 2
                else:
                    raise Exception("Unexpected winner value bug")
                    
class Selfplay:
    def __init__(self, model, controller = None, mcts_iter = args.mcts_iter):
        self.crnet = model
        if controller is None:
            self.controller = Controller()
        else:
            self.controller = controller
        self.mcts_iter = mcts_iter
        self.buffer = Buffer(model)
        self.save_in_buffer = True
        self.reset()
        
    def reset(self, state = None):
        if state is None:
            self.montecarlo_tree = MCTS({
                'array_view':np.zeros((args.M, args.N)),
                'player_turn':'red'
            }, self.crnet, self.controller)
        else:
            self.montecarlo_tree = MCTS(state, self.crnet, self.controller)
        
    def play_own_state(self):
        self.crnet.eval()
        return self.get_action_from_mcts()

    def play_given_state(self, state):
        self.crnet.eval()
        self.reset(state)
        return self.get_action_from_mcts()

    def get_action_from_mcts(self):
        if(self.mcts_iter>0):
            self.mcts_iterate(add_dirichlet = False, evaluation = True) 
            return self.__sample_action_from_policy(
                self.montecarlo_tree.get_policy_from_child_visits()
            )
        else:
            return self.__sample_action_from_policy(
                self.montecarlo_tree.root.get_policy()
            )
    
    def mcts_iterate(self, add_dirichlet = True, evaluation = False):
        
        if evaluation:
            num_iter = self.mcts_iter
        elif args.do_playout_cap:
            if np.random.rand()<args.playout_cap_p:
                num_iter = args.mcts_iter_full
                self.save_in_buffer = True
            else:
                num_iter = args.mcts_iter_fast
                add_dirichlet = False
                self.save_in_buffer = False
        else:
            num_iter = self.mcts_iter
            self.save_in_buffer = True
            
        for _ in range(num_iter):
            self.montecarlo_tree.selection(self.montecarlo_tree.root, add_dirichlet)
            
            
    def __sample_action_from_policy(self, policy):
        return np.unravel_index(
            np.random.choice(args.M*args.N, p = policy.reshape(-1)), 
            policy.shape
        )
        
    def __temperature_schedule(self, move):
        temperature = 1
        return temperature
        
    def play_turn(self, move = None):
        self.mcts_iterate()        
        policy = self.montecarlo_tree.get_policy_from_child_visits(
            self.__temperature_schedule(move)
        )
        if self.save_in_buffer:
            self.buffer.push([
                self.montecarlo_tree.root.state,
                0,
                policy
            ])
        else:
            self.buffer.push([
                None,
                0,
                policy
            ])
        action = self.__sample_action_from_policy(policy)
        return action
        
    def train_setup(self):
        self.reset()
        self.controller.reset()
        move = 1
        self.crnet.eval()
        self.crnet.to(torch.device('cpu'))
        while True:
            action = self.play_turn(move)
            eventflag = self.controller.event(action[0], action[1])
            if eventflag is None:
                self.montecarlo_tree.update(action)
            else:
                break
            move += 1
        self.buffer.assign_values()

    def train(self):
        self.train_setup()
        Trainer.train(self.buffer, self.crnet)
