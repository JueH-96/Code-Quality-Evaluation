# YOUR CODE HERE
import sys
import itertools

# Function to solve the polyomino tiling problem
def solve():
    # Read the input descriptions of the three polyominoes
    polys_raw = []
    for i in range(3):
        poly_lines = []
        for _ in range(4):
            # Read each line and remove trailing newline characters
            try:
                 line = sys.stdin.readline()
                 if not line: # Handle EOF immediately
                      print("Error reading input: Unexpected EOF", file=sys.stderr)
                      return
                 poly_lines.append(line.strip())
            except EOFError:
                 # Handle potential unexpected end of input
                 print("Error reading input: Unexpected EOF", file=sys.stderr)
                 return

        # Basic check: ensure we read 4 lines for each polyomino
        if len(poly_lines) != 4:
             print(f"Error reading input: Polyomino {i+1} description incomplete", file=sys.stderr)
             return

        polys_raw.append(poly_lines)

    # Parse the raw input into sets of coordinates for each polyomino
    # Use 0-based indexing for rows and columns (0 to 3)
    polys = []
    total_cells = 0
    for i in range(3):
        coords = set()
        for r in range(4):
            # Check if the line has the correct length
            if len(polys_raw[i][r]) != 4:
                 print(f"Error reading input: Line {r+1} of polyomino {i+1} has incorrect length {len(polys_raw[i][r])}", file=sys.stderr)
                 return

            for c in range(4):
                char = polys_raw[i][r][c]
                if char == '#':
                    coords.add((r, c))
                elif char != '.':
                     # Check for invalid characters
                     print(f"Error reading input: Invalid character '{char}' at polyomino {i+1}, row {r+1}, col {c+1}", file=sys.stderr)
                     return

        # Check if the polyomino is empty, although constraints state they are not
        # if not coords:
        #     print(f"Error: Polyomino {i+1} is empty, which violates constraints.", file=sys.stderr)
        #     return # Or handle as appropriate if constraints allowed empty pieces

        polys.append(coords)
        total_cells += len(coords)

    # First check: The total number of cells in all polyominoes must be 16 to fill the 4x4 grid
    if total_cells != 16:
        print("No")
        return

    # Precompute all unique shapes (canonical forms) for each polyomino
    # A canonical form is obtained by rotating and then normalizing (translating) the shape
    shapes = [] # shapes[i] will store a list of unique shapes for polyomino i

    for i in range(3): # Process each of the 3 polyominoes
        current_poly_coords = polys[i]
        
        unique_shapes_set = set() # Using a set avoids storing duplicate shapes

        # Generate shapes for 0, 90, 180, 270 degree rotations
        current_shape_coords = current_poly_coords
        for _ in range(4): 
            # Normalize the current shape by translating it
            # The goal is to have the minimum row index and minimum column index both be 0
            if not current_shape_coords: 
                 # This case is excluded by problem constraints (polyominoes are not empty)
                 min_r, min_c = 0, 0 
            else:
                # Find the minimum row and column indices among the cells of the shape
                min_r = min(r for r, c in current_shape_coords)
                min_c = min(c for r, c in current_shape_coords)
            
            # Create the normalized shape by shifting all coordinates
            # Represent the shape as a frozenset of tuples to make it hashable for the set
            normalized_shape = frozenset((r - min_r, c - min_c) for r, c in current_shape_coords)
            unique_shapes_set.add(normalized_shape)

            # Rotate the shape 90 degrees clockwise for the next iteration
            # The transformation rule for 90-degree clockwise rotation on a 4x4 grid is (r, c) -> (c, 3-r)
            rotated_shape_coords = set()
            for r, c in current_shape_coords:
                 rotated_shape_coords.add((c, 3 - r))
            current_shape_coords = rotated_shape_coords
        
        # Store the list of unique shapes found for this polyomino
        shapes.append(list(unique_shapes_set)) 

    # Define the recursive backtracking function to check for tilings
    # k: Index (0, 1, or 2) indicating which polyomino from the permutation p we are currently trying to place
    # p: A tuple representing the current permutation of polyomino indices (e.g., (0, 1, 2), (1, 0, 2), etc.)
    # grid: A 4x4 list of lists representing the current state of the grid (True if cell occupied, False otherwise)
    
    # Note: Memoization was attempted but removed because modifying list grid state in-place complicates memoization keys.
    # Passing immutable tuples would work with memoization but increases overhead. Sticking to modify-and-revert without memoization.
    
    def check(k, p, grid): 
        # Base case: If k reaches 3, it means all 3 polyominoes have been successfully placed
        if k == 3:
            return True 

        current_poly_idx = p[k] # Get the actual index (0, 1, or 2) of the polyomino based on permutation p

        # Iterate through all precomputed unique shapes for this polyomino
        for shape in shapes[current_poly_idx]: # shape is a frozenset of relative coordinates (r, c)
            
            # Determine the dimensions (height H, width W) of the shape's bounding box
            max_r = -1
            max_c = -1
            if shape: 
                 # For a normalized shape (min coordinates are 0,0), max coordinates define size
                 max_r = max(r for r, c in shape)
                 max_c = max(c for r, c in shape)
            else: 
                 continue # Skip if shape is somehow empty (should not happen)

            H = max_r + 1
            W = max_c + 1

            # Try placing this shape at every possible top-left position (tr, tc) on the grid
            # The loops iterate through valid translations that keep the shape within bounds
            for tr in range(4 - H + 1): # tr is the row offset (translation row)
                for tc in range(4 - W + 1): # tc is the column offset (translation column)
                    
                    can_place = True # Flag to check if the placement is valid at (tr, tc)
                    coords_to_occupy = [] # Store the absolute grid coordinates this placement would use
                    
                    # Check each cell of the shape for validity at the target position
                    for r, c in shape:
                        R, C = r + tr, c + tc # Calculate absolute grid coordinates
                        
                        # Check for overlap: If the target cell grid[R][C] is already True (occupied)
                        if grid[R][C]:
                            can_place = False # Placement invalid due to overlap
                            break
                        coords_to_occupy.append((R, C))
                    
                    # If the placement is valid (no overlaps)
                    if can_place:
                        # Tentatively place the polyomino by marking cells as occupied
                        for R, C in coords_to_occupy:
                            grid[R][C] = True
                        
                        # Recursively call 'check' to place the next polyomino (k+1)
                        if check(k + 1, p, grid):
                            return True # Success! A valid tiling has been found down this path
                        
                        # Backtrack: If the recursive call did not find a solution, undo the placement
                        for R, C in coords_to_occupy:
                            grid[R][C] = False

        # If we tried all shapes and all positions for the current polyomino p[k] and none worked
        return False

    # Main execution logic: Iterate through all possible orderings (permutations) of the three polyominoes
    poly_indices = [0, 1, 2]
    for p in itertools.permutations(poly_indices):
        # Initialize a fresh empty grid for each permutation attempt
        grid = [[False]*4 for _ in range(4)]
        # Start the backtracking search for this permutation, beginning with the first polyomino (k=0)
        if check(0, p, grid):
            print("Yes") # If check returns True, a solution is found
            return # Exit the program successfully

    # If the loop completes without finding a solution for any permutation
    print("No") # No tiling is possible

# Run the main function to solve the problem
solve()