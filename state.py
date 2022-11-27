from imports import *
'''
class for game state; stores the player's turn
stores number of orbs and color as numpy arrays, treats the color
of a vacant cell as 0, +1 if red and -1 if green.
get_array_view is the most important function here
which returns a dictionary having a board view and
showing the player's turn, which is the state representation
'''
class State:
    def __init__(self, givenstate=None):
        # board has M rows and N columns        
        # initially, it is the red player's turn
        self.turn = 'red'
        # get the degree array
        self.generate_degree_arr()
        # initialize orbs as a numpy array, initially no orbs are present
        self.orbs = np.zeros((args.M,args.N), dtype=int)
        # color is 0 if no color, 1 if red color and -1 if green color
        self.color = np.zeros((args.M,args.N), dtype=int)        
        if givenstate is not None:
            self.set_state(givenstate['array_view'], givenstate['player_turn'])
        
    def generate_degree_arr(self):
        self.degree = np.ones((args.M,args.N), dtype=int)+1
        # selecting the middle cells , M-2 rows x N-2 cols in dimensions
        self.degree[1:-1,1:-1] += 2
        # selecting the edges except the corners
        self.degree[[0,-1],1:-1] += 1
        self.degree[1:-1,[0,-1]] += 1
        
    # given a state's 2d list (or numpyarray) and the turn of the player 
    # set the state of the current object    
    def set_state(self, listview, turn):
        arr = np.array(listview)
        self.orbs = np.abs(arr)
        self.color = np.sign(arr)
        self.turn = turn
        
    def get_array_view(self):
        '''
        returns a 2d array and a string, 
        in the array, the magnitude of each entry being number of orbs, 
        positive number means red orbs and negative number means green orbs
        and string indicates the turn of the player
        '''
        return {
            'array_view':self.orbs*self.color,
            'player_turn':self.turn
        }