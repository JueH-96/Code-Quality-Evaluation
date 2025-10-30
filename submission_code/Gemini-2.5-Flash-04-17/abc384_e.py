import heapq
from collections import deque
import sys

def solve():
    # Read input values
    H, W, X = map(int, sys.stdin.readline().split())
    P, Q = map(int, sys.stdin.readline().split())

    # Adjust P, Q to 0-indexed (from 1-indexed)
    start_r, start_c = P - 1, Q - 1

    # Read the grid of slime strengths
    S = []
    for _ in range(H):
        S.append(list(map(int, sys.stdin.readline().split())))

    # Initialize Takahashi's strength with the strength of the starting slime
    current_strength = S[start_r][start_c]

    # Set to keep track of cells that have been added to the priority queue or visited (absorbed).
    # This prevents adding the same cell multiple times to the PQ.
    # A cell is considered if it's the starting cell, or if it's adjacent to an occupied cell.
    # It serves as a combined 'in_queue' and 'visited' status for the purpose of adding to PQ.
    in_pq_or_visited = set()

    # Priority queue (min-heap) storing tuples: (threshold, strength, row, col).
    # The threshold is slime_strength * X. We prioritize slimes with lower thresholds.
    # This allows us to always attempt to absorb the 'easiest' available slime first among candidates.
    pq = []

    # Queue for cells that have just become part of Takahashi's occupied region (either initial or absorbed).
    # We need to check the neighbors of these cells to find new slimes that become adjacent.
    q_explore = deque()

    # Directions for checking neighbors (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Start the process by adding the initial cell to the exploration queue and marking it as considered.
    q_explore.append((start_r, start_c))
    in_pq_or_visited.add((start_r, start_c))

    # The main loop continues as long as there are cells to explore from (meaning the occupied region grew recently)
    # OR there are candidate slimes in the priority queue that might be absorbable.
    while q_explore or pq:
        # Phase 1: Explore neighbors of cells in the exploration queue.
        # These are cells that are now part of Takahashi's occupied region.
        # Check their neighbors and add valid, unconsidered ones to the priority queue.
        # This happens before attempting absorption to ensure the set of candidates in the PQ is updated
        # based on the current occupied region.
        while q_explore:
            r, c = q_explore.popleft()

            # Check all 4 neighbors of the current cell (r, c)
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                # Check if the neighbor is within grid bounds
                if 0 <= nr < H and 0 <= nc < W:
                    # If the neighbor cell has not been considered before (neither in PQ nor visited)
                    if (nr, nc) not in in_pq_or_visited:
                        slime_strength = S[nr][nc]
                        # Calculate the threshold for absorption: slime_strength * X.
                        # Use Python's arbitrary precision integers for large values.
                        threshold = slime_strength * X

                        # Add the slime to the priority queue. The priority is the threshold.
                        # Store (threshold, strength, row, col).
                        heapq.heappush(pq, (threshold, slime_strength, nr, nc))

                        # Mark the cell as considered (added to PQ).
                        in_pq_or_visited.add((nr, nc))

        # Phase 2: Attempt to absorb the easiest available slime from the priority queue.
        # This phase runs only after exploring all current neighbors (q_explore is empty)
        # and there are candidates in the PQ.
        if pq:
            # Get the slime with the minimum threshold without removing it yet.
            # heapq[0] provides the smallest element efficiently in Python's heap implementation.
            threshold, strength, r, c = pq[0]

            # Check the absorption condition: slime_strength < current_strength / X
            # Using integer arithmetic, this is equivalent to strength * X < current_strength
            # since X >= 1. The threshold is pre-calculated as strength * X.
            if current_strength > threshold:
                # The slime is absorbable. Remove it from the priority queue.
                heapq.heappop(pq)

                # Increase Takahashi's strength by the absorbed slime's strength.
                current_strength += strength

                # The absorbed cell (r, c) is now part of Takahashi's occupied region.
                # Add it to the exploration queue so its neighbors will be checked in the next Phase 1.
                q_explore.append((r, c))

                # The cell (r,c) is already in the in_pq_or_visited set from when it was added to the PQ.
                # It is now conceptually 'visited' or 'occupied'.

            else:
                # The slime with the minimum threshold is not absorbable with the current strength.
                # Since the priority queue is sorted by threshold, no other slime in the PQ
                # (which all have thresholds >= this one) can be absorbed either *at this moment*.
                # Because current_strength only increases, slimes might become absorbable later,
                # but only if we were to absorb something else first. However, the greedy choice
                # of the minimum threshold slime ensures that if *any* slime is absorbable,
                # absorbing the one with the minimum threshold is the best step to potentially
                # unlock further absorptions. If the easiest one isn't absorbable, nothing else
                # currently in the PQ is. Thus, no more absorptions are possible from the
                # current set of candidates in the PQ.
                break # Exit the main while loop

        # The loop continues. If Phase 1 added new cells to q_explore, the next iteration starts with Phase 1.
        # If Phase 2 resulted in an absorption, it added a cell to q_explore, so the next iteration starts with Phase 1.
        # If Phase 2 failed to absorb, the 'break' statement is hit, and the loop terminates.
        # If both q_explore and pq are empty, the loop condition fails, and the loop terminates.

    # Print the final maximum strength.
    print(current_strength)

solve()