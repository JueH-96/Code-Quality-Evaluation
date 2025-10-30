# YOUR CODE HERE
import sys
import heapq

def solve():
    H, W = map(int, sys.stdin.readline().split())
    
    grid = []
    start_pos = None
    goal_pos = None
    for r in range(H):
        row_str = sys.stdin.readline().strip()
        grid.append(row_str)
        for c, char in enumerate(row_str):
            if char == 'S':
                start_pos = (r, c)
            elif char == 'T':
                goal_pos = (r, c)

    N = int(sys.stdin.readline())
    medicines_map = {}
    for _ in range(N):
        R, C, E = map(int, sys.stdin.readline().split())
        medicines_map[(R - 1, C - 1)] = E # Convert to 0-indexed coordinates

    # dist[r][c] stores the maximum energy Takahashi can have when he is at cell (r, c).
    # This energy could be from a path or boosted by a medicine at (r, c).
    # Initialize with -1 to indicate unvisited or unreachable with non-negative energy.
    dist = [[-1] * W for _ in range(H)]

    # Priority queue stores (-energy, r, c) to simulate a max-heap with heapq (which is a min-heap).
    # We want to process states with higher energy first, as higher energy allows more moves.
    pq = []

    # Initial state: Takahashi starts at 'S' with 0 energy.
    start_r, start_c = start_pos
    dist[start_r][start_c] = 0
    heapq.heappush(pq, (-0, start_r, start_c))

    # Directions for movement: (dr, dc) for Up, Down, Left, Right
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while pq:
        # Pop the state with the highest current energy
        current_energy, r, c = heapq.heappop(pq)
        current_energy = -current_energy # Convert back to positive energy value

        # If we have already found a path to (r, c) that yields equal or more energy
        # than `current_energy`, then processing this state would be suboptimal or redundant. Skip it.
        if current_energy < dist[r][c]:
            continue

        # --- Medicine Use Consideration ---
        # Check if there is a medicine at the current cell (r, c)
        if (r, c) in medicines_map:
            med_energy = medicines_map[(r, c)]
            
            # If using this medicine would give MORE energy than `current_energy`
            # AND this `med_energy` is a new maximum for `dist[r][c]`
            # (The second condition is vital to prevent unnecessary re-processing if an even
            # better energy state for this cell has already been processed or is in the queue)
            if med_energy > current_energy and med_energy > dist[r][c]:
                # Update the maximum energy achievable at this cell
                dist[r][c] = med_energy
                # Push this new, higher-energy state back into the priority queue.
                # This ensures that future paths originating from this cell with the
                # boosted energy will be prioritized and explored.
                heapq.heappush(pq, (-med_energy, r, c))
                # Crucial: If we just found a way to achieve more energy at (r,c) and re-queued it,
                # any further movements from the current `(r,c, current_energy)` state would be suboptimal.
                # So, we stop processing this lower-energy path and let the re-queued, higher-energy
                # state be handled by the priority queue later.
                continue
            # If `med_energy <= current_energy`, using the medicine wouldn't provide an energy increase.
            # We continue processing movements from this cell using `current_energy`.

        # --- Movement Consideration ---
        # If Takahashi's current energy is 0, he cannot move to an adjacent cell.
        # This check is performed AFTER medicine consideration, as a medicine might
        # provide enough energy to allow movement from a 0-energy cell.
        if current_energy == 0:
            continue # Cannot move if energy is zero

        # Explore all four adjacent cells (up, down, left, right)
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check boundaries: ensure the new cell (nr, nc) is within the grid
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            # Check for obstacles: Takahashi cannot move into '#' cells
            if grid[nr][nc] == '#':
                continue

            # Energy after moving: movement consumes 1 energy
            next_energy = current_energy - 1

            # If moving to (nr, nc) with `next_energy` results in a higher maximum energy
            # for (nr, nc) than what was previously recorded, update it and push to PQ.
            if next_energy > dist[nr][nc]:
                dist[nr][nc] = next_energy
                heapq.heappush(pq, (-next_energy, nr, nc))

    # After the Dijkstra-like search completes, check if the goal position was reachable
    # with non-negative energy. If `dist[goal_r][goal_c]` is still -1, it means the goal
    # was never reached with valid energy.
    goal_r, goal_c = goal_pos
    if dist[goal_r][goal_c] >= 0:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

solve()