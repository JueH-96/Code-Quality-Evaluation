# YOUR CODE HERE
import sys
import heapq
# from collections import defaultdict # Using defaultdict could simplify dist check slightly but standard dict is fine

# Optional: Faster I/O for large inputs
# import io, os
# input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline

def solve():
    # Read grid dimensions H (height/rows) and W (width/columns)
    H, W = map(int, sys.stdin.readline().split())
    
    F = []
    # Read grid building heights F_ij. Store in a 0-indexed list of lists.
    # F[r][c] will store the height of the building at block (r+1, c+1).
    for r in range(H):
        F.append(list(map(int, sys.stdin.readline().split())))

    # Read the number of queries Q
    Q = int(sys.stdin.readline())

    queries = []
    # Read all Q queries and store them.
    # Each query consists of start block (Ai, Bi), start floor Yi,
    # target block (Ci, Di), and target floor Zi. All are 1-based.
    for i in range(Q):
        line = list(map(int, sys.stdin.readline().split()))
        queries.append(line) 

    results = [] # List to store the answers for each query

    # Process each query one by one
    for q_idx in range(Q):
        # Unpack query parameters
        Ai, Bi, Yi, Ci, Di, Zi = queries[q_idx]

        # Initialize data structures for Dijkstra's algorithm for the current query
        
        # `dist` dictionary stores the minimum cost (number of stair uses) found so far
        # to reach each state (r, c, k). States are represented by tuples (row, col, floor).
        # Keys are state tuples, values are minimum costs. Initialized implicitly to infinity.
        dist = {} 
        
        # `pq` is the priority queue (min-heap) used by Dijkstra.
        # It stores tuples (cost, r, c, k), ordered primarily by cost.
        pq = [(0, Ai, Bi, Yi)] 
        
        # Set the distance to the initial state (start location) to 0.
        dist[(Ai, Bi, Yi)] = 0

        final_cost = -1 # Initialize the final cost for this query. Will be updated when target is reached.

        # Main Dijkstra loop: continues as long as the priority queue is not empty
        while pq:
            # Extract the state with the minimum cost from the priority queue
            cost, r, c, k = heapq.heappop(pq)

            # Current state represented as a tuple for easy dictionary key usage
            state = (r, c, k)
            
            # Optimization: If the extracted cost is greater than the already known minimum cost
            # to reach this state, skip processing. This avoids redundant work and cycles.
            # `dist.get(state, float('inf'))` safely gets the minimum cost, defaulting to infinity if state not seen.
            if cost > dist.get(state, float('inf')):
                 continue

            # Check if the current state is the target state
            if r == Ci and c == Di and k == Zi:
                final_cost = cost # Target reached, store the minimum cost
                break # Found the shortest path, can exit Dijkstra early for this query

            # --- Explore neighbors using stairs (cost increases by 1) ---
            
            # Try moving Up one floor
            # Check if moving up is possible: current floor k must be less than the building height F[r-1][c-1]
            # Note: F array is 0-indexed, while r and c are 1-based.
            if k < F[r-1][c-1]: 
                next_k = k + 1 # Floor number increases by 1
                next_state = (r, c, next_k) # State tuple for the neighbor
                new_cost = cost + 1 # Cost increases by 1 for using stairs
                
                # Relaxation step: Check if the path through the current state offers a shorter path to the neighbor state
                if new_cost < dist.get(next_state, float('inf')):
                    dist[next_state] = new_cost # Update the minimum cost to reach the neighbor state
                    heapq.heappush(pq, (new_cost, r, c, next_k)) # Add the neighbor state to the priority queue

            # Try moving Down one floor
            # Check if moving down is possible: current floor k must be greater than 1
            if k > 1: 
                next_k = k - 1 # Floor number decreases by 1
                next_state = (r, c, next_k) # State tuple for the neighbor
                new_cost = cost + 1 # Cost increases by 1 for using stairs
                
                # Relaxation step: Check if this path is shorter
                if new_cost < dist.get(next_state, float('inf')):
                    dist[next_state] = new_cost # Update minimum cost
                    heapq.heappush(pq, (new_cost, r, c, next_k)) # Add neighbor to priority queue

            # --- Explore neighbors using walkways (cost does not increase) ---
            
            # Check all 4 cardinal directions (North, South, East, West) represented by changes in row (dr) and column (dc)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                 nr, nc = r + dr, c + dc # Calculate neighbor coordinates (1-based)
                 
                 # Check if the neighbor coordinates are within the valid grid boundaries (1 to H, 1 to W)
                 if 1 <= nr <= H and 1 <= nc <= W:
                     # Check if a walkway is usable at the current floor k:
                     # The building at the neighbor block (nr, nc) must have at least k floors.
                     # The current building implicitly meets k <= F[r-1][c-1] because state (r,c,k) is valid.
                     if k <= F[nr-1][nc-1]: # F array is 0-indexed
                         next_state = (nr, nc, k) # Neighbor state: same floor k, different block (nr, nc)
                         new_cost = cost # Walkway move has 0 stair cost, so the cost remains the same
                         
                         # Relaxation step: Check if this path is shorter
                         if new_cost < dist.get(next_state, float('inf')):
                            dist[next_state] = new_cost # Update minimum cost
                            # Add the neighbor state to the priority queue. Note that 0-cost edges are handled correctly by standard Dijkstra.
                            heapq.heappush(pq, (new_cost, nr, nc, k)) 


        # After Dijkstra finishes for the query (either found target or priority queue is empty),
        # append the computed minimum cost (or -1 if target was unreachable, though implies reachable) to the results list.
        results.append(final_cost)

    # Print all computed results, one per line, corresponding to each query.
    for res in results:
        print(res)

# Call the main function to execute the solution logic
solve()