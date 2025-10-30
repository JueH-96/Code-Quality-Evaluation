import math
import itertools

def solve():
    grid = []
    for _ in range(3):
        # Read each row of the grid as space-separated integers
        grid.append(list(map(int, input().split())))

    # Define all 9 cells using 0-indexed (row, col) tuples
    all_cells = [(r, c) for r in range(3) for c in range(3)]

    # Define the 8 lines (3 horizontal, 3 vertical, 2 diagonal).
    # Each line is a tuple of 3 cell coordinates.
    lines = [
        # Horizontal lines
        ((0,0), (0,1), (0,2)),
        ((1,0), (1,1), (1,2)),
        ((2,0), (2,1), (2,2)),
        # Vertical lines
        ((0,0), (1,0), (2,0)),
        ((0,1), (1,1), (2,1)),
        ((0,2), (1,2), (2,2)),
        # Diagonal lines
        ((0,0), (1,1), (2,2)), # Top-left to bottom-right
        ((0,2), (1,1), (2,0))  # Top-right to bottom-left
    ]

    total_permutations = math.factorial(9)
    good_permutations_count = 0

    # Iterate over all possible permutations of the 9 cells.
    # Each 'p_cells' is a tuple representing the order Takahashi sees the cells.
    # For example, if p_cells[0] is (1,1), then cell (1,1) is the first cell seen.
    for p_cells in itertools.permutations(all_cells):
        
        disappointed_in_this_permutation = False
        
        # Create a mapping from cell coordinates to their position (index) in the current permutation.
        # This allows for efficient O(1) lookup of a cell's viewing order position.
        cell_to_pos = {cell: i for i, cell in enumerate(p_cells)}

        # Check each of the 8 predefined lines for the disappointment condition
        for line_cells in lines:
            s1, s2, s3 = line_cells
            
            # Get the numerical values stored in the cells of the current line
            val1 = grid[s1[0]][s1[1]]
            val2 = grid[s2[0]][s2[1]]
            val3 = grid[s3[0]][s3[1]]

            # Determine the pattern of values in this line.
            # The problem guarantees that no line will have all three values identical (e.g., X, X, X).
            # Therefore, lines are either:
            #   1. (X, X, Y) where X != Y (two values are the same, one is different)
            #   2. (X, Y, Z) where X, Y, Z are all distinct
            
            unique_val_cell = None # This variable will store the cell that holds the unique value (Y)
                                   # if the line is of type (X, X, Y).
            
            # Case 1: X, X, Y pattern
            if val1 == val2 and val1 != val3:
                unique_val_cell = s3
            elif val1 == val3 and val1 != val2:
                unique_val_cell = s2
            elif val2 == val3 and val2 != val1:
                unique_val_cell = s1
            # Case 2: X, Y, Z pattern (all distinct). 'unique_val_cell' remains None.
            # In this case, disappointment is impossible because the first two values seen will always be different.
            
            if unique_val_cell: # This block executes only if the line is of type X, X, Y
                # For an (X, X, Y) line, disappointment occurs if and only if:
                # 1. The two cells with value X are seen first and second among the three.
                # 2. The cell with value Y (unique_val_cell) is seen third (last) among the three.
                
                pos_unique = cell_to_pos[unique_val_cell]
                
                # Find the cells with value X (the "other" two cells in the line)
                other_cells = [c for c in line_cells if c != unique_val_cell]
                pos_other1 = cell_to_pos[other_cells[0]]
                pos_other2 = cell_to_pos[other_cells[1]]
                
                # Check if the unique_val_cell was indeed seen last among the three
                if pos_unique > pos_other1 and pos_unique > pos_other2:
                    disappointed_in_this_permutation = True
                    break # This permutation leads to disappointment, no need to check other lines
        
        # If no disappointment was found after checking all 8 lines for this permutation, it's a "good" one.
        if not disappointed_in_this_permutation:
            good_permutations_count += 1
            
    # Calculate the probability
    probability = good_permutations_count / total_permutations
    print(probability)