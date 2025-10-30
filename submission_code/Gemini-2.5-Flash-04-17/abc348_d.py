import heapq
import sys

# Increase recursion depth for potentially deep paths if needed, although Dijkstra is iterative.
# sys.setrecursionlimit(3000) # Default is often sufficient.

def solve():
    # Read input dimensions and grid
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Read medicine information
    N = int(sys.stdin.readline())
    medicines = {} # Store medicines as (r, c) -> energy
    for _ in range(N):
        r, c, e = map(int, sys.stdin.readline().split())
        medicines[(r - 1, c - 1)] = e # Use 0-indexed coordinates

    # Find start and goal locations
    start = (-1, -1)
    goal = (-1, -1)
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'T':
                goal = (r, c)

    # max_energy[r][c] stores the maximum energy Takahashi can have upon reaching cell (r, c)
    # Initialize with -1, indicating unreachable or not yet visited with non-negative energy.
    # We use -1 because energy can be 0 upon arrival.
    max_energy = [[-1] * W for _ in range(H)]

    # Priority Queue for Dijkstra. Stores tuples (-energy, r, c).
    # We use negative energy to make heapq work as a max-priority queue based on energy.
    # The state with the highest energy has the highest priority.
    pq = []

    # Initial state: Start at 'S' with 0 energy.
    # The max energy at the start cell is initially 0.
    max_energy[start[0]][start[1]] = 0

    # Add the initial state (S location, 0 energy) to the priority queue.
    # The medicine logic inside the loop will handle any initial boost from a medicine at S
    # when the start cell is first processed from the PQ.
    heapq.heappush(pq, (-0, start[0], start[1]))

    # Possible moves: up, down, left, right (delta row, delta column)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Dijkstra's algorithm
    while pq:
        # Pop the state (cell and energy) with the highest energy discovered so far for any cell
        # heapq pops the smallest element, so we pop the one with the largest negative energy
        e, r, c = heapq.heappop(pq)
        e = -e # Convert negative energy back to positive for calculations

        # If we have already found a way to reach (r, c) with more energy than 'e',
        # this path (leading to energy 'e') is suboptimal, so we skip processing this state.
        # This check uses the current value of max_energy[r][c], which might have been
        # updated by a better path or medicine use since this state was pushed.
        if e < max_energy[r][c]:
            continue

        # --- Process current cell (r, c) with current maximum energy e ---

        # Check if there is a medicine at the current cell (r, c)
        # In this Dijkstra model, using a medicine at (r, c) potentially provides
        # a new, higher energy level for this cell, regardless of how we arrived.
        # This simulates using the medicine if it yields the best energy seen so far for this cell.
        if (r, c) in medicines:
            med_e = medicines[(r, c)]
            # If using this medicine would result in a higher energy level than
            # the current maximum energy recorded for cell (r, c), update the max_energy
            # for this cell and add it back to the priority queue.
            # This allows subsequent exploration from (r,c) to use this boosted energy.
            if med_e > max_energy[r][c]:
                 max_energy[r][c] = med_e # Update the maximum energy for cell (r,c)
                 # Add the cell back to the priority queue with the new higher energy.
                 # This ensures that paths starting from (r,c) with this new higher energy
                 # will be explored according to Dijkstra's logic (states with higher energy
                 # are prioritized).
                 # We push negative max_energy[r][c] to maintain the max-heap behavior for energy.
                 heapq.heappush(pq, (-max_energy[r][c], r, c))
                 # Note: The moves explored below in this iteration still use the energy 'e'
                 # that was just popped from the PQ. If the medicine boosted max_energy[r][c],
                 # the state with the higher medicine energy will be processed in a future
                 # iteration when it's popped from the PQ according to its higher priority.

        # The energy available for movement from the current cell (r, c) is the
        # maximum energy achieved *upon arriving* at (r, c) via the current path (energy 'e').
        # If a medicine was used and boosted max_energy[r][c], that higher value will be
        # used when the cell is re-popped from the PQ in a later iteration.
        current_effective_energy = e # Use the energy value popped from PQ for moves

        # Takahashi cannot move from this cell if the current energy is 0 or less.
        # He needs strictly positive energy to move to an adjacent cell.
        if current_effective_energy <= 0:
            continue # Skip exploring neighbors if energy is insufficient for movement

        # Explore valid adjacent cells (neighbors)
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check if the neighbor cell (nr, nc) is within the grid boundaries
            # and is not an obstacle '#'.
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                # Moving to an adjacent cell consumes 1 energy.
                new_energy = current_effective_energy - 1

                # If reaching cell (nr, nc) via this path results in a higher
                # energy level than the maximum previously recorded for (nr, nc),
                # update max_energy[nr][nc] and add (nr, nc) to the priority queue.
                # We only add to PQ if we found a better path (in terms of energy).
                if new_energy > max_energy[nr][nc]:
                    max_energy[nr][nc] = new_energy # Update max energy for (nr, nc)
                    # Push the neighbor state (cell and its new energy level) to the priority queue.
                    # Push negative new_energy for max-heap behavior.
                    heapq.heappush(pq, (-new_energy, nr, nc))

    # After the Dijkstra search completes (priority queue is empty),
    # check if the goal cell (tr, tc) is reachable with non-negative energy.
    # Reaching a cell means arriving at it. Energy must be >= 0 upon arrival.
    # The value max_energy[goal[0]][goal[1]] stores the maximum energy achieved
    # upon arriving at the goal cell through any explored path.
    if max_energy[goal[0]][goal[1]] >= 0:
        print("Yes")
    else:
        print("No")

# Read input from standard input and run the solver function
solve()