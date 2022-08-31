import matplotlib.pyplot as plt

# wins = [67.6, 77.8, 83.4, 88.3]
wins = [19, 20, 21, 22, 21, 20, 22, 25, 20, 21, 23, 16, 23, 20, 23]

generations = [i for i in range(1, 16)]
print(generations)

# font size of the labels
plt.rcParams.update({'font.size': 18})
# plotting the points
plt.plot(generations, wins, 'bo-')

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
plt.ylabel('Wins')
  
# giving a title to my graph
plt.title('Number of wins across generations')
  
# function to show the plot
plt.show()