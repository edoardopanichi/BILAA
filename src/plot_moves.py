import matplotlib.pyplot as plt

# moves = [38.16, 34.94, 35.84, 37.26]
moves = [42.5, 45, 41, 40, 31.4, 39.2, 33, 32.4, 36.4, 29.4, 41.6, 38.6, 36.6, 36.8, 38.6]
		
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