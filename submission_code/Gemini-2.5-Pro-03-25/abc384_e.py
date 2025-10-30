# YOUR CODE HERE
import heapq
import sys

# Function to solve the problem
def solve():
    # Read grid dimensions H, W and factor X from standard input
    H, W, X = map(int, sys.stdin.readline().split())
    
    # Read starting position P, Q (1-based indexing) from standard input
    P, Q = map(int, sys.stdin.readline().split())
    # Convert P, Q to 0-based index for list/array access
    P -= 1 
    Q -= 1

    # Read slime strengths for each cell into a 2D list S
    S = []
    for i in range(H):
        # Read a row of strengths, convert to integers, and append to S
        S.append(list(map(int, sys.stdin.readline().split())))

    # Initialize Takahashi's current strength with the strength of the slime in the starting cell
    current_strength = S[P][Q]
    
    # Keep track of visited/absorbed cells using a 2D boolean array.
    # `visited[r][c] = True` means the slime at cell (r, c) has been absorbed.
    visited = [[False for _ in range(W)] for _ in range(H)]
    # Mark the starting cell as visited since Takahashi starts there.
    visited[P][Q] = True

    # Initialize a min-priority queue (min-heap) using Python's heapq module.
    # It will store tuples: (required_strength, row, col).
    # `required_strength` is the minimum strength Takahashi needs *before* absorbing the slime at (row, col).
    # This threshold is calculated as `X * S[row][col] + 1`. 
    # This calculation stems from the condition S_adj < S_T / X, which is equivalent to X * S_adj < S_T (for positive X).
    # To satisfy this strict inequality with integer strengths, Takahashi's strength S_T must be at least X * S_adj + 1.
    pq = [] 

    # Define possible moves to adjacent cells (4 directions: right, left, down, up)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Add the initial neighbors of the starting cell (P, Q) to the priority queue.
    # These are the first potential slimes Takahashi could absorb.
    for dr, dc in moves:
        nr, nc = P + dr, Q + dc
        
        # Check if the neighbor coordinates (nr, nc) are within the grid boundaries [0, H-1] x [0, W-1].
        if 0 <= nr < H and 0 <= nc < W:
             # Calculate the required strength for Takahashi to absorb this neighbor slime.
             # Python's integers handle arbitrary size, supporting large strengths and X values.
            required_strength = X * S[nr][nc] + 1
            # Push the tuple (required_strength, neighbor_row, neighbor_col) onto the min-heap.
            # heapq maintains the heap property based on the first element of the tuple, `required_strength`.
            heapq.heappush(pq, (required_strength, nr, nc))

    # Main loop: continues as long as the priority queue is not empty.
    # An empty PQ means there are no more reachable, unabsorbed slimes adjacent to Takahashi's region.
    while pq:
        # Peek at the element with the minimum required strength currently in the PQ.
        # `pq[0]` accesses the smallest element (root of the min-heap) without removing it.
        req_S, r, c = pq[0] 

        # Check if Takahashi's current strength (`current_strength`) is sufficient 
        # to absorb the slime at (r, c), which requires at least `req_S` strength.
        if current_strength >= req_S:
            # If strength is sufficient, we can potentially absorb this slime.
            # First, remove this slime candidate from the priority queue using heappop.
            heapq.heappop(pq)

            # Before absorbing, double-check if this cell has already been visited/absorbed.
            # This is necessary because a cell might have been added to the PQ multiple times
            # if it became adjacent to the growing Takahashi region from different absorbed neighbors.
            # If it's already visited, we simply skip it and the loop continues to the next smallest element in PQ.
            if visited[r][c]:
                continue

            # If the cell (r, c) hasn't been visited:
            # Absorb the slime:
            # 1. Increase Takahashi's total strength by adding the strength of the absorbed slime S[r][c].
            current_strength += S[r][c]
            # 2. Mark this cell (r, c) as visited, indicating it's now part of Takahashi.
            visited[r][c] = True

            # After absorbing slime at (r, c), its own neighbors become potentially absorbable if not already visited.
            # Add its unvisited neighbors to the priority queue.
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                
                # Check if the neighbor is within grid bounds and has not been visited yet.
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                    # Calculate the required strength for this new neighbor slime.
                    new_req_strength = X * S[nr][nc] + 1
                    # Add this neighbor to the priority queue. It will be considered later based on its required strength.
                    heapq.heappush(pq, (new_req_strength, nr, nc))
        
        else:
            # If Takahashi's `current_strength` is LESS than the minimum `required_strength` (`req_S`)
            # found at the top of the priority queue:
            # This means Takahashi cannot absorb the slime (r, c). Since (r, c) has the minimum requirement
            # among all currently adjacent slimes available in the PQ, he cannot absorb any other available slime either 
            # (as others would require strength >= req_S).
            # Therefore, the absorption process must terminate.
            break

    # After the loop finishes (either because the PQ becomes empty or no more slimes can be absorbed),
    # `current_strength` holds the maximum possible strength Takahashi could achieve.
    # Print the final accumulated strength to standard output.
    print(current_strength)

# Call the solve function to execute the logic when the script is run.
solve()