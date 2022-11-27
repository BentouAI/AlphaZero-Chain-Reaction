from imports import *
from state import *

class Node:
    def __init__(self, state, model):
        # saves state as a dictionary
        self.state = state
        # needs access to the neural network model
        self.model = model

        # W is the total reward and N is the number of playouts
        self.W = 0
        self.N = 0

        self.value = None
        self.policy = None
        
        # sets which actions are valid and which are invalid
        # in the variables self.valid_actions and 
        # self.invalid_actions respectively
        self.set_action_validity()

        # for all valid_actions, initialize new nodes (but don't 
        # fill them yet with states)
        self.initialize_edges()
        self.win = None

    def initialize_edges(self):
        if self.state is not None:
            self.children = {}
            for row in range(args.M):
                for col in range(args.N):
                    if self.valid_actions[row][col]:
                        self.children[(row,col)] = Node(None, self.model)

    def set_action_validity(self):
        if self.state is not None:
            if self.state['player_turn'] == 'red':
                self.invalid_actions = self.state['array_view']<0
            else:
                self.invalid_actions = self.state['array_view']>0
            self.valid_actions = ~self.invalid_actions

    def make_forward_pass(self):
        with torch.no_grad():
            out = self.model(
                self.model.state_array_view_to_tensor(self.state)
            )
        self.policy = out['policy'][0].cpu().numpy()
        self.value = out['value'].cpu().item()    
        self.policy[self.invalid_actions] = 0
        
        # handling rare case where sum becomes zero 
        # can happen because of treating low magnitude values as zero
        if self.policy.sum()==0:
            self.policy[self.valid_actions] = 1
        
        self.policy /= self.policy.sum()
        
    def get_policy(self):
        if self.policy is None:
            self.make_forward_pass()
        return self.policy
    
    def get_value(self):
        if self.value is None:
            self.make_forward_pass()
        return self.value

class MCTS:
    # state is a dictionary containing array_view and player_turn
    def __init__(self, state, model, controller):
        self.root = Node(state, model)
        self.model = model
        self.controller = controller
        
    def get_puct_val(self, parent, child, prior):
        
        # child.W indicates the total rewards of the 
        # "opposite player". The child of a red player 
        # is green and vice versa. The total rewards must be negated 
        # while calculating action value pair for the current player
        # because positive rewards for the opponent are negative rewards
        # for us.

        q_val = 0 if child.N==0 else -child.W/child.N    
        # alphazero puct formula
        puct_val = q_val + args.cpuct * prior * (np.sqrt(parent.N))/(1+child.N)
        
        return puct_val
        
    def add_dirichlet_noise(self, node):
        num_valid_actions = node.valid_actions.sum()
        noise_vector = np.random.dirichlet([args.dirichlet_alpha]*num_valid_actions)
        noise_arr = np.zeros((args.M,args.N))
        noise_arr[node.valid_actions] = noise_vector
        return noise_arr

    def select_best_child(self, node, is_root):
        prior = node.get_policy()

        if is_root:
            noise_arr = self.add_dirichlet_noise(node)
            prior = prior * (1-args.epsilon) + noise_arr * args.epsilon
            
        max_puct_val = None
        best_child = None
        best_action = None
        
        for action, child in node.children.items():
            puct_val = self.get_puct_val(node, child, prior[action])
            if max_puct_val is None or puct_val > max_puct_val:
                max_puct_val = puct_val
                best_child = child
                best_action = action
            
        return best_child, best_action
    
    # backward passes of results from MCTS is handled through recursion
    def selection(self, node, root = False, logging = False, actions = None):
        
        best_child, best_action = self.select_best_child(node, root)

        if(logging):
            if actions == None:
                actions = []
            actions.append(best_action[0]*args.M+best_action[1])

        if best_child.state is None:
            val = self.expand_and_evaluate(node, best_action, best_child)

            if(logging):
                with open('data.txt','a') as fout:
                    fout.write(', '.join([str(i) for i in actions]))
                    fout.write('\n')

        else:
            val = self.selection(best_child, False, logging, actions)
        
        node.W += val
        node.N += 1
        
        '''
        value for the current player positively correlates with the likelihood of the current player 
        winning. If the current player has value +1, it means it would win, 
        which implies its children (and its parent, if it is not the root node), who are actually
        the opposite player, would have value -1.
        '''
        return -val
        
    def expand_and_evaluate(self, parent, action, child):
        if child.win is None:
            next_state_obj, win = self.controller.get_next_state(parent.state, action)
            next_state = next_state_obj.get_array_view()
            if win is None:
                child.state = next_state
                child.set_action_validity()
                child.initialize_edges()
                val = child.get_value()
            else:
                child.win = win
                val = -1 if parent.state['player_turn'] == win else 1 
        else:
            val = -1 if parent.state['player_turn'] == child.win else 1

        child.W += val
        child.N += 1

        return -val 
            
    def get_policy_from_child_visits(self, temperature = 1):
        result = np.zeros((args.M,args.N))
        for action, child in self.root.children.items():
            result[action] = child.N ** (1/temperature)
        
        result = result/result.sum()
        return result
    
    def update(self, action):
        self.root = self.root.children[action]