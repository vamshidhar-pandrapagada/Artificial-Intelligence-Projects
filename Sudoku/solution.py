assignments = []


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [row+col for row in A for col in B]

def encode_board():
    """
    This function is used to encode the board
    row_units: Set of 9 boxes for each row. Any number between 1-9 can appear only once in the row unit
    col_units: Set of 9 boxes for each column. Any number between 1-9 can appear only once in the col unit
    square_units: Set of 9 boxes for each 3x3 square. Any number between 1-9 can appear only once in the square units    
    diagnol_units: Set of 9 boxes for 2 diagnols of the board. Any number between 1-9 can appear only once in the diagnol_units
    
    return list of all units
    """
    row_units = [cross(r, cols) for r in rows]
    col_units = [cross(rows, c) for c in cols]
    square_units = [cross(r, c) for r in ('ABC','DEF','GHI') for c in ('123','456','789')]
    
    diagnol_lr = [row+col for row,col in zip(rows,cols)]
    diagnol_rl = [row+col for row,col in zip(rows,reversed(cols))]
    diagnol_units = [diagnol_lr,diagnol_rl]
    
    return [row_units,col_units,square_units,diagnol_units]

# We use rows and columns to encode the SUDOKU board. Hence, we'll define rows and columns as GLOBAL variables
rows = 'ABCDEFGHI'
cols = '123456789'
boxes = cross(rows, cols)
all_units = encode_board()

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    
    grid_dict = {}
    grid_dict = {box:val for box,val in zip(boxes,grid)}
    for box,val in grid_dict.items():
        if (grid_dict[box]=='.'):
            grid_dict = assign_value(grid_dict, box, '123456789')    
    return grid_dict
    

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    
   new_values = values.copy()   
   
   for units in all_units[0:3]: #Iterate over row_units, column_units and Square_units
       for box,val in values.items():
           if (len(values[box]) == 1):               
               #scan all the row units and eliminate the values from peers corresponding to each box
               eliminate_from_unit = [unit for unit in units if box in unit] #Collect all boxes from the unit if any box in the unit has only 1 value
               eliminate_from_unit = [retain_box for retain_box in eliminate_from_unit[0] if box!= retain_box] #Remove box with 1 value
               for b in eliminate_from_unit:
                   new_values = assign_value(new_values,b,new_values[b].replace(val,''))
   
      
   
    
   """
   SOLVE DIAGNOL SUDOKU
   Step 1: Select all the boxes corresponding to the 2 diagols on the board. Let's call these 2 units as Diagnol units. 
           Diagnol unit selection is done in encode_board function and its result is used here   
   Step 2: For every box in the diagnol unit, eliminate the values that can't appear on the box, based on its peers. 
           Then Apply Only choice technique if there is only one box in diagnol unit which would allow a certain digit.  
   Step 3: Repeat Steps 1 and 2 to propagate through all the boxes in both the diagnol units until all constraints are cleared.
   """
    
    
    #Logic to eliminate from Diagnols 
   diagnol_units = all_units[3]
   final_values = new_values.copy()
   
   for dg_unit in diagnol_units:
       for diagnol_box in dg_unit:
           if (len(new_values[diagnol_box]) == 1):  
               #scan all the diagnol units and eliminate the values from peers corresponding to each box
               eliminate_from_unit = [retain_box for retain_box in dg_unit if diagnol_box!= retain_box] #Remove box with 1 value
               for b in eliminate_from_unit:
                   final_values = assign_value(final_values,b,final_values[b].replace(new_values[diagnol_box],''))    
   
        
   return final_values   
    
    

def only_choice(values):
    
    unitlist = all_units[0] + all_units[1] + all_units[2] + all_units[3]
    for unit in unitlist:
        for num in '123456789':
            replace_places = [box for box in unit if num in values[box]]
            if (len(replace_places) == 1):
                values = assign_value(values,replace_places[0],num) 
    
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Step 1: The first step is to pick all boxes in the grid which have values of length 2.   
    Step 2: Pick any of the boxes selected in Step 1 and check if the box has a twin in its constraint space 
           (in this case, space is called as a unit). If Yes, apply eliminate to remove all occurences of the twin's values 
           from the unit space. This technique reduces the overall search space while solving the sudoku.  
           Also ELIMINATE applied in this step may introduce constraints on other parts of the board.  
    Step 3: Repeat Steps 1 and 2 to clear all the naked twin constraints on the board until eliminate is applied on 
            all the units corresponding to the boxes selected in Step1. This is called constraint propagatation.  
    
    
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
           Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    
    new_values = values.copy()   
    
    # Find all instances of length 2 boxes
    length2boxes = {box:val for box,val in new_values.items() if len(new_values[box])==2}
    
    for box,val in length2boxes.items():
        
        #Find if naked twin exists in any of the units (row, col, square, diagnol)
        matched_units = [unit for unit_type in all_units for unit in unit_type for b in unit if b == box ]        
        # Eliminate the naked twins as possibilities for their peers
        for units in matched_units:
            twin_found = [r for r in units if values[r] == val]
            if (len(twin_found) > 1):
                boxes_to_replace = [boxes for boxes in units if boxes not in twin_found]
                for r in boxes_to_replace:
                    for v in val:
                        new_values = assign_value(new_values,r,new_values[r].replace(v,'')) 
   
    return new_values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        
        # Use the Eliminate rule
        values = eliminate(values)
        
        # Use the Only Choice rule
        values = only_choice(values)
        
        # Use the Naked Twin rule
        values = naked_twins(values)
        
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    values = reduce_puzzle(values)
    if values is False:
        return False
    solved_or_not = len([box for box in values.keys() if len(values[box]) != 1])
    if (solved_or_not == 0):
        return values
    
    # Choose one of the unfilled squares with the fewest possibilities
    unfilled_sq = [k for k in sorted(values, key=lambda k: len(values[k])) if len(values[k])>1][0]
    
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for val in values[unfilled_sq]:
        new_sudoku = values.copy()
        new_sudoku[unfilled_sq] = val
        
        depth_first_call = search(new_sudoku)
        if (depth_first_call):
            return depth_first_call

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    #Step 1: Convert the Grid to a form of Sudoku Dictionary. Key = Box Name, Value = Value of the box
    sudoku = grid_values(grid)
    
    #Step 2: Use Depth First Search to solve the sudoku
    #Depth First Search uses reduce Puzzle function to perform Constraint Propagation by combining eliminate and only_choice functions.
    sudoku = search(sudoku)
    
    
    solved_or_not = len([box for box in sudoku.keys() if len(sudoku[box]) != 1])
    if (solved_or_not == 0):
        return sudoku
    else:
        return False  #No solution exists
    
    
    

if __name__ == '__main__':
    #diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    
    diag_sudoku_grid ='.5.......6.3..24...7.1....38.4.....7.........3.....2.97....1.2...96..7.1.......4.'
    """
    Tested on on other diagnal SUDOKUs
    diag_sudoku_grid = '.45...63.2...1...59..8.5..7..9...3...3.....7...8...5..8..5.3..15...2...3.26...95.'
    
    Hard Diagnol Sudokus
    diag_sudoku_grid ='.5.......6.3..24...7.1....38.4.....7.........3.....2.97....1.2...96..7.1.......4.'
    diag_sudoku_grid ='...1.6...3...5...1....7....4...9...5.157.239.7...3...2....8....6...1...7...9.7...'
    """
    
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
