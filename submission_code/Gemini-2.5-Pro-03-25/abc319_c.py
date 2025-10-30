# YOUR CODE HERE
import sys
import itertools
import math

# Function to calculate cell coordinates (row, col) from index k (0-8)
def k_to_coords(k):
    """Converts cell index k (0-8) to (row, col) coordinates."""
    # Integer division gives row, modulo gives column for a 3x3 grid
    return (k // 3, k % 3)

def solve():
    """Reads the 3x3 grid, identifies potentially disappointing lines (PDLs),
    iterates through all permutations of viewing the cells, counts the 'good' permutations
    (those that don't cause disappointment), and prints the probability."""
    
    # Read 3x3 grid input from standard input
    grid = []
    for _ in range(3):
        grid.append(list(map(int, sys.stdin.readline().split())))

    # Define the 8 lines (3 rows, 3 columns, 2 diagonals) by their cell coordinates
    lines = []
    # Rows
    for r in range(3):
        lines.append([(r, 0), (r, 1), (r, 2)])
    # Columns
    for c in range(3):
        lines.append([(0, c), (1, c), (2, c)])
    # Diagonals
    lines.append([(0, 0), (1, 1), (2, 2)]) # Main diagonal
    lines.append([(0, 2), (1, 1), (2, 0)]) # Anti-diagonal

    # Identify Potentially Disappointing Lines (PDLs)
    # A PDL is a line where two cells have the same value A, and one cell has a different value B.
    # The problem guarantees that no line has all three cells with the same value.
    PDLs = []
    for line in lines:
        cell1, cell2, cell3 = line
        # Get values from the grid using cell coordinates
        val1 = grid[cell1[0]][cell1[1]]
        val2 = grid[cell2[0]][cell2[1]]
        val3 = grid[cell3[0]][cell3[1]]
        
        # Count occurrences of each value in the line
        vals = [val1, val2, val3]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        
        # A line is a PDL if there are exactly two distinct values.
        # Because the problem guarantees no line has all same values (count=1),
        # len(counts) == 2 implies the counts must be {2, 1}.
        if len(counts) == 2: 
           # Identify the cell coordinate that holds the unique value (the one with count 1)
           unique_val_cell_coord = None
           
           # Find which value appears only once
           val_with_count_one = -1
           for v, count in counts.items():
                if count == 1:
                    val_with_count_one = v
                    break # Found the unique value

           # Find which cell holds this unique value
           if val1 == val_with_count_one:
               unique_val_cell_coord = cell1
           elif val2 == val_with_count_one:
               unique_val_cell_coord = cell2
           else: # val3 must be == val_with_count_one
               unique_val_cell_coord = cell3
           
           # Store the line coordinates and the coordinate of the unique value cell
           PDLs.append({'line': line, 'unique_cell': unique_val_cell_coord})

    # Count the number of 'good' permutations (those not causing disappointment)
    good_permutations_count = 0
    
    # Total number of cells is 9. Iterate through all permutations of cell indices 0 to 8.
    cell_indices = list(range(9))
    for p_indices in itertools.permutations(cell_indices):
        # p_indices is a tuple representing the order cells are revealed, e.g., (0, 3, 1, ...)
        # where each number is an index from 0 to 8 mapping to a cell.
        
        # Create a mapping from cell coordinate (r, c) to its position (0-8) in the current permutation.
        # This allows efficient lookup of when a cell is revealed.
        coord_to_pos = {}
        for k in range(9): # k is the position (0-8) in the permutation sequence
            # p_indices[k] is the cell index (0-8) revealed at position k
            # k_to_coords converts cell index to (r, c) coordinates
            coord_to_pos[k_to_coords(p_indices[k])] = k

        is_bad = False # Flag to track if the current permutation causes disappointment
        # Check each identified PDL against the current permutation order
        for pdl_info in PDLs:
            line_coords = pdl_info['line']           # List of 3 coordinates [(r1,c1), (r2,c2), (r3,c3)] for the PDL
            unique_cell_coord = pdl_info['unique_cell'] # Coordinate (r,c) of the cell with the unique value in this PDL

            cell1, cell2, cell3 = line_coords # The three cell coordinates in the line
            
            # Get the positions (indices 0-8 in the permutation) when these three cells are revealed
            pos1 = coord_to_pos[cell1]
            pos2 = coord_to_pos[cell2]
            pos3 = coord_to_pos[cell3]
            
            # Find the maximum position among the three cells; this corresponds to the cell revealed last.
            last_pos_idx = max(pos1, pos2, pos3)
            
            # Determine the coordinates of the cell that was revealed last in this line.
            # p_indices[last_pos_idx] gives the cell index (0-8) seen last.
            # k_to_coords converts this index back to (r, c) coordinates.
            last_cell_coord = k_to_coords(p_indices[last_pos_idx])

            # Takahashi gets disappointed if the cell revealed last is the one with the unique value for this line.
            if last_cell_coord == unique_cell_coord:
                is_bad = True # Mark this permutation as 'bad'
                break # No need to check other PDLs for this permutation; it's already bad.

        # If the permutation did not cause disappointment for any PDL, it's a 'good' permutation.
        if not is_bad:
            good_permutations_count += 1

    # Calculate the probability: (number of good permutations) / (total number of permutations)
    total_permutations = math.factorial(9) # Total permutations of 9 items is 9!
    
    # Handle division by zero just in case, although N=9 means total_permutations is 362880 > 0.
    if total_permutations == 0:
        probability = 0.0
    else:
        probability = good_permutations_count / total_permutations
    
    # Print the final probability, formatted to 15 decimal places for sufficient precision.
    print(f"{probability:.15f}")

# Execute the solve function when the script is run
solve()