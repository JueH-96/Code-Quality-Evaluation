# YOUR CODE HERE
import sys
from collections import deque

# Set higher recursion depth limit if using recursive DFS, although BFS is used here and avoids deep recursion issues.
# sys.setrecursionlimit(2000*1000) # BFS avoids deep recursion, so this usually isn't necessary.

def solve():
    # Read input dimensions H (height/rows) and W (width/columns)
    H, W = map(int, sys.stdin.readline().split())
    # Read the grid configuration
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Define the modulus
    M = 998244353

    # Modular exponentiation function (calculates base^power % M)
    def fast_pow(base, power):
        """ Computes (base^power) % M using binary exponentiation. """
        result = 1
        # Ensure base is already modulo M
        base %= M 
        while power > 0:
            # If power is odd, multiply result with base
            if power % 2 == 1:
                result = (result * base) % M
            # Square the base and halve the power
            base = (base * base) % M
            power //= 2
        return result

    # Modular inverse function using Fermat's Little Theorem
    def mod_inverse(a):
        """ Computes modular inverse of a modulo M. """
        # Based on Fermat's Little Theorem: a^(M-2) = a^(-1) (mod M) for prime M.
        # M = 998244353 is prime.
        # The problem constraints guarantee NR >= 1.
        # Since NR <= H*W <= 1000*1000 = 1,000,000 < M, NR is not divisible by M.
        # Thus, the modular inverse exists.
        return fast_pow(a, M - 2)

    # --- Step 1: Find initial green components and assign component IDs ---
    
    # visited array to keep track of visited cells during BFS
    visited = [[False for _ in range(W)] for _ in range(H)]
    # comp_id array stores the component ID for each cell. 
    # 0 indicates red or unvisited green cell initially.
    comp_id = [[0 for _ in range(W)] for _ in range(H)] 
    
    C0 = 0 # Initial count of green connected components
    current_comp_idx = 1 # Component IDs start from 1 for clarity
    
    q = deque() # Queue for Breadth-First Search (BFS)

    # Iterate through each cell in the grid
    for r in range(H):
        for c in range(W):
            # If cell is green ('#') and has not been visited yet (means it's part of a new component)
            if S[r][c] == '#' and not visited[r][c]:
                C0 += 1 # Increment component count
                q.append((r, c)) # Add the starting cell to the BFS queue
                visited[r][c] = True # Mark it as visited
                comp_id[r][c] = current_comp_idx # Assign the current component ID
                
                # Perform BFS starting from (r, c) to find all cells in this component
                while q:
                    curr_r, curr_c = q.popleft() # Get the next cell from the queue
                    
                    # Check its 4 adjacent neighbors (up, down, left, right)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        # Check if the neighbor coordinates are within the grid bounds
                        if 0 <= nr < H and 0 <= nc < W:
                            # Check if the neighbor cell is green and has not been visited
                            if S[nr][nc] == '#' and not visited[nr][nc]:
                                visited[nr][nc] = True # Mark neighbor as visited
                                comp_id[nr][nc] = current_comp_idx # Assign the same component ID
                                q.append((nr, nc)) # Add neighbor to the queue for further exploration
                
                # After BFS completes for this component, increment the component ID for the next one
                current_comp_idx += 1 

    # --- Step 2: Calculate Sum of K values for Red Cells ---
    
    total_K = 0 # Initialize the sum of K_rc values
    NR = 0 # Initialize the count of red cells ('.')

    # Iterate through each cell in the grid again
    for r in range(H):
        for c in range(W):
            # If the cell is red ('.')
            if S[r][c] == '.':
                NR += 1 # Increment red cell count
                neighbor_comp_ids = set() # Use a set to store unique component IDs of adjacent green cells
                
                # Check 4 adjacent neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    
                    # Check if neighbor is within grid bounds
                    if 0 <= nr < H and 0 <= nc < W:
                        # If neighbor is green ('#')
                        if S[nr][nc] == '#':
                           # Add its component ID to the set. 
                           # Component IDs for green cells are stored in comp_id[nr][nc] and are > 0.
                           neighbor_comp_ids.add(comp_id[nr][nc]) 
                
                # K_rc is the number of distinct green components adjacent to this red cell
                K_rc = len(neighbor_comp_ids)
                
                # Add K_rc to the total sum
                total_K += K_rc

    # --- Step 3: Calculate Expected Value Modulo M ---
    
    # The problem statement guarantees NR >= 1 (at least one red cell exists).
    # Therefore, division by NR is well-defined and NR is not zero.
    
    # Calculate the modular inverse of NR
    NR_inv = mod_inverse(NR)

    # The expected value E is given by E = C0 + 1 - (sum(K_rc) / NR)
    # We compute this modulo M: E_mod = (C0 + 1 - (total_K * NR_inv)) % M

    # Calculate the term (total_K * NR_inv) mod M
    # total_K can potentially be large, but Python handles large integers automatically.
    # The intermediate products are taken modulo M.
    term = (total_K * NR_inv) % M
    
    # Calculate (C0 + 1) mod M
    # C0 is at most H*W, which fits within standard integer types.
    C0_plus_1 = (C0 + 1) % M

    # Calculate the final expected value: (C0 + 1 - term) mod M
    # Add M before taking the final modulo to ensure the result is non-negative
    # because (C0 + 1 - term) might be negative before the modulo operation.
    expected_value = (C0_plus_1 - term + M) % M 

    # Print the final result
    print(expected_value)

# Call the main function to solve the problem
solve()