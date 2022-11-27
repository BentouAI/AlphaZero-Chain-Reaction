from imports import *

def reflect_sec_diag(X):
    return np.fliplr(np.rot90(X))

class Buffer(Dataset):
    MAX = 15000
    def __init__(self, model):
        self.model = model
        self.reset()
    
    def reset(self):
        self.data = []
        self.temp_buffer = []

    def assign_values(self):
        assign = 1
        index = len(self.temp_buffer) - 1
        while index>=0:
            self.temp_buffer[index][1] = assign
            assign *= -1
            index -= 1
        self.__augment()
        self.temp_buffer = []
    
    def __apply_func(self, element, func, *args):
        if func is not None:
            S = func(element[0]['array_view'], *args)
            V = element[1]
            P = func(element[2], *args)
        else:
            S = element[0]['array_view']
            V = element[1]
            P = element[2]
        return [self.model.state_array_view_to_tensor(
            {
                'array_view':S,
                'player_turn':element[0]['player_turn']
            }
        ),V,P]
    
    def __augment(self):
        for element in self.temp_buffer:
            if element[0] is not None:
                self.__push(self.__apply_func(element, None))
                self.__push(self.__apply_func(element, np.flip, 0))
                self.__push(self.__apply_func(element, np.flip, 1))
                self.__push(self.__apply_func(element, np.rot90))
                self.__push(self.__apply_func(element, np.rot90, 2))
                self.__push(self.__apply_func(element, np.rot90, 3))
                self.__push(self.__apply_func(element, np.transpose))
                self.__push(self.__apply_func(element, reflect_sec_diag))
            
    def __push(self, element):
        self.data.append(element)
        if len(self.data) > Buffer.MAX:
            self.data.pop(0)
            
    def push(self, element):
        self.temp_buffer.append(element)
            
    def extend(self, datalist):
        self.data.extend(datalist)
        L = len(self.data)
        if L > Buffer.MAX:
            self.data = self.data[L-Buffer.MAX:]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'state':self.data[idx][0],
            'value':self.data[idx][1],
            'policy':self.data[idx][2]
        }
    
    def find_number_of_unique_states(self):
        K = set([tuple(self.data[i][0].reshape(-1).tolist()) for i in range(len(self.data))])
        return len(K)