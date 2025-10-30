# YOUR CODE HERE
import sys

# Function to calculate Manhattan distance between two points (r1, c1) and (r2, c2)
def manhattan_distance(r1, c1, r2, c2):
    """Calculates the Manhattan distance between two points."""
    return abs(r1 - r2) + abs(c1 - c2)

def solve():
    # Read dimensions H, W and distance D from input
    # H: number of rows, W: number of columns, D: maximum Manhattan distance for humidification
    H, W, D = map(int, sys.stdin.readline().split())
    
    # Read the grid layout from standard input
    # S is a list of strings representing the grid rows
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Find all floor cells (represented by '.') and store their coordinates
    # using 0-based indexing for rows and columns.
    floor_cells = []
    for r in range(H):
        for c in range(W):
            if S[r][c] == '.':
                floor_cells.append((r, c))

    # Get the total number of floor cells
    num_floor_cells = len(floor_cells)

    # Problem constraints guarantee that there are at least two floor cells (num_floor_cells >= 2).
    # This means we can always choose two distinct floor cells to place humidifiers.
    # If the constraints were different, we might need to handle cases where num_floor_cells < 2.
    # Example: if num_floor_cells < 2: print(0); return

    # Precompute step: For each potential humidifier location `k` (which must be a floor cell),
    # determine the set of floor cells that would be humidified if a humidifier were placed there.
    # We store the *indices* of these humidified floor cells in a set for efficiency.
    # `humidified_indices_by_source[k]` will store the set of indices `j` such that
    # `floor_cells[j]` is within Manhattan distance D from `floor_cells[k]`.
    humidified_indices_by_source = []
    for k in range(num_floor_cells):
        # Get the coordinates of the k-th floor cell (potential humidifier location)
        r_k, c_k = floor_cells[k]
        
        # Initialize a set to store the indices of floor cells humidified by this source `k`
        current_humidified_indices = set()
        
        # Iterate through all floor cells `j` to check if they are within distance D from source `k`
        for j in range(num_floor_cells):
            # Get the coordinates of the j-th floor cell
            r_j, c_j = floor_cells[j]
            
            # Calculate the Manhattan distance between floor cell `j` and source `k`
            # Check if this distance is less than or equal to D
            if manhattan_distance(r_k, c_k, r_j, c_j) <= D:
                # If within range, add the index `j` to the set for source `k`
                current_humidified_indices.add(j) 
        
        # Append the computed set of humidified indices for source `k` to our list
        humidified_indices_by_source.append(current_humidified_indices)

    # Initialize the variable to keep track of the maximum number of humidified floor cells found
    max_humidified_count = 0
    
    # Main loop: Iterate through all distinct pairs of floor cell indices (idx1, idx2)
    # These indices represent the locations where the two humidifiers are placed.
    # We use nested loops structure `for idx1 in range(N): for idx2 in range(idx1 + 1, N):`
    # where N = num_floor_cells. This efficiently iterates over all unique pairs (idx1, idx2)
    # ensuring idx1 < idx2, thus avoiding duplicate pairs and pairs where idx1 == idx2.
    for idx1 in range(num_floor_cells):
        for idx2 in range(idx1 + 1, num_floor_cells): 
            # Retrieve the precomputed sets of humidified floor cell indices for the selected pair of sources
            set1 = humidified_indices_by_source[idx1]
            set2 = humidified_indices_by_source[idx2]
            
            # Compute the union of the two sets. The union represents all unique floor cells
            # that are humidified by at least one of the two humidifiers placed at idx1 and idx2.
            union_set = set1.union(set2)
            
            # The size (length) of the union set gives the total count of unique humidified floor cells
            # for this specific pair of humidifier locations.
            current_pair_humidified_count = len(union_set)
            
            # Update the overall maximum count if the count for the current pair is greater
            # than the maximum count found so far.
            max_humidified_count = max(max_humidified_count, current_pair_humidified_count)

    # After checking all pairs, max_humidified_count holds the maximum possible number.
    # Print the final result to standard output.
    print(max_humidified_count)

# Call the main function `solve()` to execute the logic.
solve()