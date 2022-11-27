from imports import *

# pygame graphics code part        
'''
Creates the pygame window
Does drawing functions like drawing the grid using rectangles
and drawing orbs using circles
The most important function here is the gameloop function
which takes the stateview dictionary and updates the pygame display.
(If you're familiar with p5.js, the gameloop function is like draw,
and __init__ is like setup,
except that we would manually call the gameloop function in a loop)
'''
class Visualizer:
    def __init__(self):
        self.cell_edge = 50
        self.width = args.N * self.cell_edge
        self.height = args.M * self.cell_edge
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chain Reaction")
        # radius of orbs
        self.R = 15
        
    def draw_grid(self, color):
        for i in range(args.M):
            for j in range(args.N):
                pygame.draw.rect(
                    self.display, 
                    color, 
                    ((j * self.cell_edge, i * self.cell_edge), (self.cell_edge,self.cell_edge)),
                    1
                )
        
    def gameloop(self, stateview):
        self.display.fill(0)
    
        if stateview['player_turn'] == 'red':
            self.draw_grid((255,0,0))
        else:
            self.draw_grid((0,200,0))

        for i in range(args.M):
            for j in range(args.N):
                Matrix = stateview['array_view']

                if Matrix[i][j]!=0:
                    if Matrix[i][j]>0:
                        color = (255,0,0)
                    else:
                        color = (0,200,0)

                    # center of the cell i,j 
                    cX = j * self.cell_edge + self.cell_edge//2
                    cY = i * self.cell_edge + self.cell_edge//2
                    
                    if abs(Matrix[i][j])==1:
                        pygame.draw.circle(self.display, color, (cX, cY), self.R)
                    elif abs(Matrix[i][j])==2:
                        pygame.draw.circle(self.display, color, (cX - self.R//2, cY), self.R)
                        pygame.draw.circle(self.display, color, (cX + self.R//2, cY), self.R)
                    elif abs(Matrix[i][j])==3:
                        pygame.draw.circle(self.display, color, (cX, cY - self.R//2),self.R)
                        pygame.draw.circle(self.display, color, (cX + self.R//2, cY + self.R//2), self.R)
                        pygame.draw.circle(self.display, color, (cX - self.R//2, cY + self.R//2), self.R)

        pygame.display.update()