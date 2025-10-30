# YOUR CODE HERE
import sys
import collections

# Function to read input, solve problem, and print output
def solve():
    # Read the size of the grid
    N = int(sys.stdin.readline())
    # Read the grid configuration
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Find the initial positions of the two players
    player_pos = []
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'P':
                player_pos.append((r, c))

    # Assign initial positions to p1_init and p2_init
    p1_init = player_pos[0]
    p2_init = player_pos[1]

    # To potentially reduce the state space and handle symmetry, we ensure that
    # the state always stores the player positions such that the first player's
    # coordinates are lexicographically less than or equal to the second player's.
    # Player position tuples (r, c) are compared lexicographically based on row then column.
    if p1_init > p2_init:
        p1_init, p2_init = p2_init, p1_init
    
    # The state is represented as a tuple (r1, c1, r2, c2), where (r1, c1) are coordinates
    # of the first player (lexicographically smaller or equal initial pos) and (r2, c2) for the second.
    # We concatenate the two position tuples to form the state tuple.
    initial_state = p1_init + p2_init 

    # Initialize a deque for BFS. Stores tuples of (state, distance).
    q = collections.deque([(initial_state, 0)])
    # Keep track of visited states using a set for O(1) average time complexity lookups.
    # This prevents cycles and redundant explorations.
    visited = {initial_state}

    # Define the four possible directions of movement: Up, Down, Left, Right
    # Represented by changes in row and column (dr, dc).
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

    # Start the Breadth-First Search
    while q:
        # Dequeue the current state and its distance from the initial state
        current_state, dist = q.popleft()
        
        # Extract player coordinates from the state tuple
        r1, c1, r2, c2 = current_state

        # Check if the goal state is reached: both players are on the same cell.
        if r1 == r2 and c1 == c2:
            # If yes, print the minimum distance (number of moves) and exit.
            print(dist)
            return 

        # Explore neighbors by trying each of the four possible moves (directions)
        for dr, dc in directions:
            
            # Calculate the potential next position for player 1 after attempting the move.
            nr1, nc1 = r1 + dr, c1 + dc
            next_r1, next_c1 = r1, c1 # Default: player stays put if the move is invalid.
            # Check if the target cell is within grid boundaries (0 <= row < N, 0 <= col < N)
            # and is not an obstacle ('#'). Note that '.' and 'P' cells are considered empty for movement.
            if 0 <= nr1 < N and 0 <= nc1 < N and grid[nr1][nc1] != '#':
                # If the move is valid, update player 1's position.
                next_r1, next_c1 = nr1, nc1 

            # Calculate the potential next position for player 2 similarly.
            nr2, nc2 = r2 + dr, c2 + dc
            next_r2, next_c2 = r2, c2 # Default: player stays put if the move is invalid.
            # Check validity of the move for player 2.
            if 0 <= nr2 < N and 0 <= nc2 < N and grid[nr2][nc2] != '#':
                 # If the move is valid, update player 2's position.
                 next_r2, next_c2 = nr2, nc2 

            # Define the next positions as tuples for easier handling.
            next_p1 = (next_r1, next_c1)
            next_p2 = (next_r2, next_c2)

            # Canonicalize the next state representation: ensure the first player's coordinates
            # are lexicographically less than or equal to the second player's coordinates.
            # This ensures consistent state representation.
            if next_p1 > next_p2:
                next_p1, next_p2 = next_p2, next_p1
            
            # Form the next state tuple by concatenating the canonical position tuples.
            next_state = next_p1 + next_p2 

            # If this resulting state has not been visited before:
            if next_state not in visited:
                # Mark it as visited.
                visited.add(next_state)
                # Enqueue the new state with the distance incremented by 1.
                q.append((next_state, dist + 1))

    # If the queue becomes empty and we haven't returned a distance, it means
    # the goal state (players on the same cell) is unreachable from the initial state.
    # In this case, print -1 as required.
    print("-1")

# Standard boilerplate to call the solve function when the script is executed.
if __name__ == '__main__':
    solve()