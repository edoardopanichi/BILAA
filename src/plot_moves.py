import matplotlib.pyplot as plt

# moves = [38.16, 34.94, 35.84, 37.26]
moves = [41.75, 41.5, 38.7, 38.8, 35.5, 36.3, 34.1, 32.9, 37, 32.7, 39.2, 40, 39.65, 32.8, 38.8]

		
generations = [i for i in range(1, 16)]

# font size of the labels
plt.rcParams.update({'font.size': 18})
# plotting the points
plt.plot(generations, moves, 'ro-')

if False: # activate if wanted
    # zip joins x and y coordinates in pairs
    for x,y in zip(generations, moves):

        label = "{:.2f}".format(y)

        plt.annotate(label, # this is the text
                    (x+0.04,y-0.1), # these are the coordinates to position the label
                    textcoords = "offset points", # how to position the text
                    xytext = (0, 10), # distance from text to points (x,y)
                    ha = 'left') # horizontal alignment can be left, right or center
    
# naming the x axis
plt.xlabel('Generations')
# naming the y axis
plt.ylabel('Number of moves')
  
# giving a title to my graph
plt.title('Number of moves before Stockfish wins')
  
# function to show the plot
plt.show()