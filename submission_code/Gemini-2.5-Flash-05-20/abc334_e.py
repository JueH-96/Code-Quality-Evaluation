import sys
import collections

def solve():
    # Read H and W from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid rows
    grid_str = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Define the modulus for calculations
    MOD = 998244353
    
    # Define directions for BFS (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]
    
    # Step 1: Initial component analysis
    # component_map[r][c] will store the ID of the component cell (r,c) belongs to.
    # If cell (r,c) is red, component_map[r][c] will remain None.
    component_map = [[None for _ in range(W)] for _ in range(H)]
    
    # Counter for initial green components
    initial_green_components = 0
    
    # visited array for the initial BFS to identify components
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    # List to store coordinates of all red cells.
    # We will iterate through these later.
    red_cells = [] 
    
    # Iterate through the grid to find initial green components and collect red cells
    for r in range(H):
        for c in range(W):
            if grid_str[r][c] == '.':
                red_cells.append((r, c))
            elif grid_str[r][c] == '#' and not visited[r][c]:
                # Found an unvisited green cell, this starts a new connected component
                initial_green_components += 1
                
                # Use a deque for efficient BFS (append and popleft)
                q = collections.deque([(r, c)])
                visited[r][c] = True
                component_map[r][c] = initial_green_components # Assign component ID
                
                while q:
                    curr_r, curr_c = q.popleft()
                    
                    for i in range(4): # Check 4 neighbors (up, down, left, right)
                        nr, nc = curr_r + dr[i], curr_c + dc[i]
                        
                        # Check bounds, if neighbor is green ('#') and unvisited
                        if 0 <= nr < H and 0 <= nc < W and \
                           grid_str[nr][nc] == '#' and not visited[nr][nc]:
                            visited[nr][nc] = True
                            component_map[nr][nc] = initial_green_components
                            q.append((nr, nc))
    
    # Step 2: Calculate total sum of components for all possible repainting scenarios
    total_sum_components = 0
    N_red = len(red_cells) # Total number of red cells
    
    # Iterate through each red cell that could be repainted
    for r, c in red_cells:
        # This set will store the distinct component IDs of green cells adjacent to (r,c)
        adjacent_comp_ids = set() 
        
        for i in range(4): # Check 4 neighbors of the current red cell (r,c)
            nr, nc = r + dr[i], c + dc[i]
            
            # If neighbor is within bounds and is an initial green cell
            if 0 <= nr < H and 0 <= nc < W and grid_str[nr][nc] == '#':
                # Add its component ID to the set.
                # component_map[nr][nc] will contain the integer ID of the component.
                adjacent_comp_ids.add(component_map[nr][nc])
        
        # Calculate the number of green components if (r,c) is repainted
        # Formula: current_components = initial_green_components - (len(adjacent_comp_ids) - 1)
        # This formula handles all cases:
        # - If len(adjacent_comp_ids) == 0: current_components = C_0 - (0 - 1) = C_0 + 1 (new component)
        # - If len(adjacent_comp_ids) == 1: current_components = C_0 - (1 - 1) = C_0 (merges into existing)
        # - If len(adjacent_comp_ids) > 1: current_components = C_0 - (k - 1) (k components merge into 1)
        current_components = initial_green_components - (len(adjacent_comp_ids) - 1)
        
        # Add this scenario's component count to the total sum, modulo MOD
        total_sum_components = (total_sum_components + current_components) % MOD
            
    # Step 3: Compute expected value modulo MOD
    # Expected value = (total_sum_components / N_red) mod MOD
    # This is equivalent to (total_sum_components * N_red_inv) mod MOD,
    # where N_red_inv is the modular multiplicative inverse of N_red.
    # Fermat's Little Theorem: a^(MOD-2) = a^(-1) (mod MOD) for prime MOD.
    
    inv_N_red = pow(N_red, MOD - 2, MOD) 
    
    # Final answer
    ans = (total_sum_components * inv_N_red) % MOD
    
    # Print the result to standard output
    sys.stdout.write(str(ans) + "
")

# Call the solve function to run the program
if __name__ == '__main__':
    solve()