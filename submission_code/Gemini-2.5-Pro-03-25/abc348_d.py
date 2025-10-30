# YOUR CODE HERE
import heapq
import sys

# Using sys.stdin.readline for faster input reading
input = sys.stdin.readline

def solve():
    # Read grid dimensions H (height/rows) and W (width/columns)
    H, W = map(int, input().split())
    
    # Read the grid layout
    grid = [input().strip() for _ in range(H)]
    
    start_pos = None
    goal_pos = None
    
    # Find start ('S') and goal ('T') positions within the grid
    # The coordinates are stored as (row, column) tuples using 0-based indexing.
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'T':
                goal_pos = (r, c)
            # Small optimization: stop searching once both S and T are found
            if start_pos and goal_pos: break
        if start_pos and goal_pos: break

    # Read the number of medicines
    N = int(input())
    
    # Store medicine information in a dictionary.
    # Key: (row, column) tuple representing medicine location (0-based index)
    # Value: energy E provided by the medicine
    medicines = {}
    for _ in range(N):
        # Read medicine location (R, C) and energy E
        R, C, E = map(int, input().split())
        # Convert from 1-based input index to 0-based internal index
        R -= 1
        C -= 1
        
        # Problem statement guarantees medicines are not at obstacles '#'.
        # If multiple medicines exist at the same location, we only care about
        # the one providing the maximum energy, as Takahashi would logically choose it.
        if (R, C) not in medicines or E > medicines[(R, C)]:
            medicines[(R, C)] = E

    # Initialize a 2D list `max_energy` to store the maximum energy Takahashi
    # can possess upon arriving at cell (r, c). Initialize all values to -1,
    # signifying that cells are initially unreachable or reached without energy.
    max_energy = [[-1] * W for _ in range(H)]

    # Initialize a priority queue `pq`. We use Python's `heapq` module, which implements
    # a min-heap. To simulate a max-heap (prioritizing states with higher energy),
    # we store the negative of the energy value.
    # Each element in the queue is a tuple: (-energy, row, column)
    pq = []

    # Initial condition check: Takahashi starts at 'S' with 0 energy.
    # He must immediately use a medicine at 'S' to gain energy and move.
    # Check if there is a medicine at the start position.
    if start_pos in medicines:
        start_energy = medicines[start_pos]
        # Set the initial maximum energy at the start position
        max_energy[start_pos[0]][start_pos[1]] = start_energy
        # Push the initial state onto the priority queue
        heapq.heappush(pq, (-start_energy, start_pos[0], start_pos[1]))
    else:
        # If there's no medicine at the start position 'S', Takahashi has 0 energy
        # and cannot make any move. The goal 'T' is unreachable (unless S == T, 
        # which is disallowed by problem constraints).
        print("No")
        return

    # Main loop of the Dijkstra-like algorithm. It explores states (cells) based on
    # maximum achievable energy.
    while pq:
        # Extract the state (cell and energy) with the highest energy from the priority queue.
        neg_curr_e, r, c = heapq.heappop(pq)
        curr_e = -neg_curr_e # Convert energy back to its positive value

        # Stale state check: If the energy `curr_e` associated with the popped state
        # is less than the maximum energy already known (`max_energy[r][c]`) for this cell (r, c),
        # it means we have already found a better path to this cell. Skip processing this state.
        # This check is crucial for correctness and efficiency.
        if curr_e < max_energy[r][c]:
            continue

        # Medicine usage check: Check if the current cell (r, c) contains a medicine.
        # The problem allows Takahashi to use a medicine at his current cell.
        # Using a medicine sets his energy to a fixed value E_i.
        # Consider using the medicine only if it provides *strictly greater* energy than
        # the current maximum recorded energy for this cell (`max_energy[r][c]`).
        # This check explores if using the medicine opens up new possibilities (higher energy state).
        if (r, c) in medicines:
            med_e = medicines[(r, c)]
            if med_e > max_energy[r][c]: 
                 # Update the maximum energy for this cell
                 max_energy[r][c] = med_e
                 # Push a new state onto the priority queue representing the situation *after*
                 # using the medicine. This state starts from (r, c) with energy `med_e`.
                 heapq.heappush(pq, (-med_e, r, c))
                 # Note: The energy `curr_e` of the state we popped is NOT modified here.
                 # The exploration of moves (below) uses `curr_e`, which represents the energy
                 # upon arrival *before* potentially using the medicine. The state pushed here
                 # `(-med_e, r, c)` handles the alternative branch where the medicine IS used.

        # Movement exploration: If Takahashi has positive energy (`curr_e > 0`), he can move
        # to adjacent cells. Each move consumes 1 energy.
        if curr_e > 0:
            # Define possible moves: right, left, down, up
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc # Calculate neighbor cell coordinates

                # Check if the neighbor cell (nr, nc) is within the grid boundaries
                # and is not an obstacle ('#').
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    # Calculate the energy Takahashi would have after moving to the neighbor
                    neighbor_e = curr_e - 1
                    
                    # If this path reaches the neighbor cell (nr, nc) with more energy than
                    # any path found so far (`max_energy[nr][nc]`), then this is a potentially
                    # better path to the neighbor.
                    if neighbor_e > max_energy[nr][nc]:
                        # Update the maximum energy recorded for the neighbor cell
                        max_energy[nr][nc] = neighbor_e
                        # Push the neighbor state onto the priority queue for further exploration
                        heapq.heappush(pq, (-neighbor_e, nr, nc))

    # Final check: After the Dijkstra-like algorithm completes (priority queue is empty),
    # we check if the goal cell 'T' was ever reached. Reachability is determined by
    # checking if the maximum energy recorded for the goal position (`max_energy[goal_pos[0]][goal_pos[1]]`)
    # is non-negative (>= 0). Even reaching the goal with exactly 0 energy counts as success.
    if max_energy[goal_pos[0]][goal_pos[1]] >= 0:
        print("Yes")
    else:
        print("No")

# Ensures the solve function is called when the script is executed directly.
if __name__ == '__main__':
    solve()