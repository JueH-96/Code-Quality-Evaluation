import heapq
import sys

def solve():
    H, W, X = map(int, sys.stdin.readline().split())
    P, Q = map(int, sys.stdin.readline().split())
    
    # Read grid strengths
    grid_strengths = []
    for _ in range(H):
        grid_strengths.append(list(map(int, sys.stdin.readline().split())))

    # Convert P, Q to 0-indexed coordinates
    start_r, start_c = P - 1, Q - 1

    # Initialize Takahashi's strength with the slime at his starting position
    current_takahashi_strength = grid_strengths[start_r][start_c]
    
    # Keep track of which slimes have been absorbed
    is_absorbed = [[False for _ in range(W)] for _ in range(H)]
    is_absorbed[start_r][start_c] = True # Takahashi's initial slime is considered absorbed
    
    # Min-priority queue: (threshold_to_absorb, slime_actual_strength, r, c)
    pq = [] 

    # Helper function to add valid, unabsorbed neighbors of a cell (r_coord, c_coord) to PQ
    def add_neighbors_to_pq(r_coord, c_coord):
        for dr_offset, dc_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Up, Down, Left, Right
            nr, nc = r_coord + dr_offset, c_coord + dc_offset
            
            if 0 <= nr < H and 0 <= nc < W: # Check boundary
                if not is_absorbed[nr][nc]: # Check if not absorbed
                    slime_val = grid_strengths[nr][nc]
                    threshold = slime_val * X
                    heapq.heappush(pq, (threshold, slime_val, nr, nc))

    # Add initial neighbors of Takahashi's starting cell to PQ
    add_neighbors_to_pq(start_r, start_c)
            
    while pq:
        threshold, slime_strength_val, r, c = heapq.heappop(pq)

        if is_absorbed[r][c]: # Already absorbed
            continue

        # Condition: slime_strength_val * X < current_takahashi_strength
        # This is: threshold < current_takahashi_strength
        if threshold < current_takahashi_strength:
            current_takahashi_strength += slime_strength_val
            is_absorbed[r][c] = True
            
            # Add neighbors of the newly absorbed slime to PQ
            add_neighbors_to_pq(r, c)
        else:
            # If smallest threshold slime cannot be absorbed, none other in PQ can.
            break
            
    sys.stdout.write(str(current_takahashi_strength) + "
")

solve()