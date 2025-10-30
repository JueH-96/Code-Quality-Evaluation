# YOUR CODE HERE
import sys

# Set higher recursion depth limit if needed (not needed for iterative find)
# Using iterative find functions avoids potential RecursionError for deep paths.
# sys.setrecursionlimit(400010) 

def solve():
    # Read input dimensions and query count
    H, W, Q = map(int, sys.stdin.readline().split())

    # Using 1-based indexing for cells (r, c) matches problem statement
    
    # Use a set to keep track of cells that currently have walls.
    # Initializing with all cells having walls.
    has_wall = set()
    # This initialization takes O(H*W) time.
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            has_wall.add((r, c))

    # Initialize total wall count
    total_walls = H * W

    # Parent dictionaries for Union-Find structures.
    # We use four separate UF structures, one for each direction (up, down, left, right).
    # The keys are coordinates (r, c) as tuples.
    # The values are the parent coordinates (pr, pc) or None if linked to boundary.
    parent_up = {}
    parent_down = {}
    parent_left = {}
    parent_right = {}

    # --- Iterative Find Functions with Path Compression ---
    # These functions find the root representative of a component in the corresponding UF structure.
    # The root represents the first potential cell with a wall encountered when moving in that direction.
    # Path compression is applied to optimize future lookups.
    # They return the coordinate tuple (r, c) of the root, or None if the search path leads out of the grid bounds.

    def find_up(r, c):
        # Base case: Out of bounds upwards
        if r < 1: return None
        
        curr = (r, c)
        # Initialize the node in UF structure if it hasn't been seen before (lazy initialization)
        if curr not in parent_up:
            parent_up[curr] = curr 
            return curr

        # Traverse the path to find the root
        path = []
        while True:
            parent = parent_up[curr]
            if parent == curr: # Found the root (node points to itself)
                root = curr
                break
            if parent is None: # Path leads to boundary marker (None)
                root = None
                break
            # Standard UF find step: Record path and move to parent
            path.append(curr)
            curr = parent
        
        # Path compression: Make all nodes in the traversed path point directly to the root
        for node in path:
            parent_up[node] = root
        return root

    def find_down(r, c):
        # Base case: Out of bounds downwards
        if r > H: return None
        curr = (r, c)
        if curr not in parent_down:
            parent_down[curr] = curr
            return curr

        path = []
        while True:
            parent = parent_down[curr]
            if parent == curr:
                root = curr
                break
            if parent is None:
                root = None
                break
            path.append(curr)
            curr = parent
        
        for node in path:
            parent_down[node] = root
        return root

    def find_left(r, c):
        # Base case: Out of bounds leftwards
        if c < 1: return None
        curr = (r, c)
        if curr not in parent_left:
            parent_left[curr] = curr
            return curr

        path = []
        while True:
            parent = parent_left[curr]
            if parent == curr:
                root = curr
                break
            if parent is None:
                root = None
                break
            path.append(curr)
            curr = parent
        
        for node in path:
            parent_left[node] = root
        return root

    def find_right(r, c):
        # Base case: Out of bounds rightwards
        if c > W: return None
        curr = (r, c)
        if curr not in parent_right:
            parent_right[curr] = curr
            return curr

        path = []
        while True:
            parent = parent_right[curr]
            if parent == curr:
                root = curr
                break
            if parent is None:
                root = None
                break
            path.append(curr)
            curr = parent
        
        for node in path:
            parent_right[node] = root
        return root

    # --- Destroy Wall Function ---
    # This function handles the destruction of a wall at (r, c).
    # It updates the `has_wall` set, decrements `total_walls`, and updates the UF structures.
    # Updating UF structures involves setting the parent pointers of the destroyed cell (r, c)
    # to effectively "skip" this cell in future searches. It connects (r, c) to the component
    # found by starting a search from the adjacent cell in each respective direction.
    def destroy_wall(r, c):
        nonlocal total_walls # Declare intent to modify the variable from the outer scope
        
        # If wall doesn't exist at (r, c) (e.g., already destroyed), do nothing.
        if (r, c) not in has_wall: return 
        
        # Remove wall status and decrement count
        has_wall.remove((r, c))
        total_walls -= 1 

        # Union logic: Link the destroyed cell (r, c) into the chain of destroyed cells or boundary.
        # For each direction, find the root starting from the adjacent cell.
        # Set the parent pointer of (r, c) in the respective UF structure to this root.
        # The root can be None if the adjacent cell is outside the grid or its chain leads to the boundary.
        
        root_above = find_up(r - 1, c) 
        parent_up[(r, c)] = root_above 

        root_below = find_down(r + 1, c) 
        parent_down[(r, c)] = root_below 

        root_left = find_left(r, c - 1) 
        parent_left[(r, c)] = root_left 

        root_right = find_right(r, c + 1) 
        parent_right[(r, c)] = root_right

    # --- Query Processing Loop ---
    for _ in range(Q):
        # Read query coordinates
        Rq, Cq = map(int, sys.stdin.readline().split())
        target_coord = (Rq, Cq)

        # Case 1: A wall exists at the target coordinate (Rq, Cq).
        if target_coord in has_wall:
            # Destroy the wall at (Rq, Cq).
            destroy_wall(Rq, Cq)
        
        # Case 2: No wall exists at the target coordinate (Rq, Cq).
        else:
            # Find the nearest walls in the four cardinal directions.
            # Use a set to collect unique coordinates of walls to be destroyed.
            to_destroy = set()
            
            # Find nearest wall upwards: Start search from (Rq-1, Cq).
            root_up = find_up(Rq - 1, Cq)
            # If a root cell is found (not None) and it currently has a wall:
            if root_up is not None and root_up in has_wall:
                 # Add this wall's coordinate to the set of walls to destroy.
                 to_destroy.add(root_up)
            
            # Find nearest wall downwards: Start search from (Rq+1, Cq).
            root_down = find_down(Rq + 1, Cq)
            if root_down is not None and root_down in has_wall:
                 to_destroy.add(root_down)

            # Find nearest wall leftwards: Start search from (Rq, Cq-1).
            root_left = find_left(Rq, Cq - 1)
            if root_left is not None and root_left in has_wall:
                 to_destroy.add(root_left)

            # Find nearest wall rightwards: Start search from (Rq, Cq+1).
            root_right = find_right(Rq, Cq + 1)
            if root_right is not None and root_right in has_wall:
                 to_destroy.add(root_right)

            # Destroy all identified walls.
            # Iterate through the unique coordinates collected.
            for r_d, c_d in to_destroy:
                 # Call destroy_wall for each coordinate. 
                 # The function itself checks if the wall still exists before acting.
                 destroy_wall(r_d, c_d) 

    # After processing all queries, print the final count of remaining walls.
    print(total_walls)

# Execute the solver function to run the program.
solve()