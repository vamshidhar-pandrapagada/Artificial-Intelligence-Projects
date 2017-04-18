# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: In Naked twins problem, We use constraint propagation to search all the local constraints in the naked twins's space to remove the numerical values that are locked to the twin. 
Twin's space here is the entire unit and Constraint is any box in the unit. Using Constraint propagation is a very effective technique which can dramatically reduce the number of possibilities and narrow down the total search space of the algorithm. In this case, once Naked twins are handled, the next iteration in the algorithm will have fewer possibilites to clear.

Below are the detailed steps implemented in the code:
* Step 1: The first step is to pick all boxes in the grid which have values of length 2.   
* Step 2: Pick any of the boxes selected in Step 1 and check if the box has a twin in its constraint space (in this case, space is called as a unit). If Yes, apply eliminate to remove all occurrences of the twin's values from the unit space. This technique reduces the overall search space while solving the sudoku.  Also ELIMINATE applied in this step may introduce constraints on other parts of the board.  
* Step 3: Repeat Steps 1 and 2 to clear all the naked twin constraints on the board until eliminate is applied on all the units corresponding to the boxes selected in Step1. This is called constraint propagation.  

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: In diagonal sudoku problem, our local constraint is any box which is part of the diagonal unit. We simply propagate through every constraint (or box) and clear it using Eliminate and Only Choice Techniques.
In this problem, Constraint space is the diagonal unit, and constraint is any box in the unit. We have already seen how constraint propagation helps us solve this problem very effeciently.

Detailed steps implemented in the code:
* Step 1: Select all the boxes corresponding to the 2 diagonals on the board. Let's call these 2 units as Diagnol units.    
* Step 2: For every box in the diagonal unit, eliminate the values that can't appear on the box, based on its peers. Then Apply Only choice technique if there is only one box in diagnol unit which would allow a certain digit.  
* Step 3: Repeat Steps 1 and 2 to propagate through all the boxes in both the diagonal units until all constraints are cleared.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

