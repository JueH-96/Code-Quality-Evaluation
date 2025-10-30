# YOUR CODE HERE
import sys
from collections import deque

def solve():
    """
    Solves the two-player grid problem using Breadth-First Search (BFS).
    This function reads the grid, finds the players' initial positions,
    and performs a BFS on the state space to find the minimum number of
    moves for the players to meet on the same cell.
    """
    # Read the size of the grid and the grid layout from standard input.
    try:
        N = int(sys.stdin.readline())
        grid = [sys.stdin.readline().strip() for _ in range(N)]
    except (IOError, ValueError):
        # This handles cases of malformed or empty input.
        print(-1)
        return

    # Find the initial positions of the two players.
    players = []
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'P':
                players.append((r, c))

    p1_start, p2_start = players

    # --- BFS Setup ---
    # A state is (r1, c1, r2, c2).
    # To treat states like (p1_pos, p2_pos) and (p2_pos, p1_pos) as identical,
    # we normalize the state by sorting the player coordinates lexicographically.
    if p1_start > p2_start:
        p1_start, p2_start = p2_start, p1_start
    
    initial_state = (*p1_start, *p2_start)

    # The queue stores tuples of (state, distance).
    q = deque([(initial_state, 0)])
    
    # The visited set stores states to avoid cycles.
    visited = {initial_state}
    
    # Define the four cardinal directions.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def try_move(r, c, dr, dc):
        """
        Calculates the new position of a player after a move attempt.
        Returns the new coordinates if the move is valid (within bounds and not an obstacle),
        otherwise returns the original coordinates.
        """
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != '#':
            return nr, nc
        return r, c

    # --- Main BFS Loop ---
    while q:
        (r1, c1, r2, c2), dist = q.popleft()

        # Goal check: If both players are at the same cell.
        if r1 == r2 and c1 == c2:
            print(dist)
            return

        # Explore all four possible moves from the current state.
        for dr, dc in directions:
            # Both players attempt to move simultaneously.
            nr1, nc1 = try_move(r1, c1, dr, dc)
            nr2, nc2 = try_move(r2, c2, dr, dc)

            p1_new, p2_new = (nr1, nc1), (nr2, nc2)
            
            # Normalize the new state before checking if it's visited.
            if p1_new > p2_new:
                p1_new, p2_new = p2_new, p1_new

            new_state = (*p1_new, *p2_new)

            # If this state has not been visited, add it to the queue and visited set.
            if new_state not in visited:
                visited.add(new_state)
                q.append((new_state, dist + 1))

    # If the queue becomes empty, the goal is unreachable.
    print(-1)

# Run the solution.
solve()