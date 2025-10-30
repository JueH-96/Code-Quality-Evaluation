import heapq
import sys

def solve():
    H, W, X = map(int, sys.stdin.readline().split())
    P, Q = map(int, sys.stdin.readline().split())
    
    S = []
    for _ in range(H):
        S.append(list(map(int, sys.stdin.readline().split())))

    # Adjust P, Q to 0-based indexing
    P_idx, Q_idx = P - 1, Q - 1

    current_takahashi_strength = S[P_idx][Q_idx]

    # Set to keep track of cells Takahashi has absorbed.
    # This acts as the "visited" set for the cells that are part of Takahashi's body.
    absorbed_cells = set()
    absorbed_cells.add((P_idx, Q_idx))

    # Priority queue stores (threshold_strength, -slime_strength, r, c).
    # The negative slime_strength is used to prioritize higher strength slimes
    # when their absorption threshold_strength values are equal (min-heap property).
    pq = []

    # Set to keep track of cells that have been added to the priority queue.
    # This prevents adding duplicate cells to the PQ, which would waste memory and time.
    # A cell is added to this set when it first becomes adjacent to Takahashi's body
    # and is a candidate for absorption. The starting cell is implicitly 'visited'
    # in this context since its neighbors are the first ones added to the PQ.
    visited_for_pq_addition = set()
    visited_for_pq_addition.add((P_idx, Q_idx))

    # Directions for neighbors (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Helper function to add valid neighbors of a newly absorbed cell to the PQ
    def add_neighbors_to_pq(r, c):
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            
            # Check boundaries and if the cell is neither absorbed nor already pending in PQ
            if 0 <= nr < H and 0 <= nc < W and \
               (nr, nc) not in absorbed_cells and \
               (nr, nc) not in visited_for_pq_addition:
                
                slime_val = S[nr][nc]
                # Calculate the minimum strength Takahashi needs to absorb this slime:
                # S_slime < current_strength / X  =>  S_slime * X < current_strength
                # Since all strengths are integers, current_strength must be at least S_slime * X + 1.
                threshold = slime_val * X + 1
                
                heapq.heappush(pq, (threshold, -slime_val, nr, nc))
                visited_for_pq_addition.add((nr, nc))

    # Populate the PQ initially with neighbors of Takahashi's starting position
    add_neighbors_to_pq(P_idx, Q_idx)

    # Main loop: process slimes from the priority queue
    while pq:
        threshold, neg_slime_S, r, c = heapq.heappop(pq)
        slime_S = -neg_slime_S

        # If Takahashi's current strength is less than the required threshold for this slime,
        # he cannot absorb it. Since the PQ is sorted by threshold (and then by -strength),
        # he also cannot absorb any other slimes currently in the PQ (as their thresholds
        # will be equal or higher). Therefore, we break the loop as no more slimes can be absorbed.
        if current_takahashi_strength < threshold:
            break
        
        # If strength is sufficient, absorb the slime
        current_takahashi_strength += slime_S
        absorbed_cells.add((r, c))

        # Add neighbors of the newly absorbed slime to the PQ, expanding Takahashi's reach
        add_neighbors_to_pq(r, c)

    # Print the maximum possible strength of Takahashi
    sys.stdout.write(str(current_takahashi_strength) + "
")

solve()