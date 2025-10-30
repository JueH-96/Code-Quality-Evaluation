# YOUR CODE HERE
import sys
import collections

def get_next_pos(r, c, dr, dc, grid, N):
    """Calculates the next position for a player."""
    nr, nc = r + dr, c + dc
    # Check if the destination is within bounds and not an obstacle ('#')
    if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != '#':
        return (nr, nc)
    else:
        # If the destination is invalid, the player stays put
        return (r, c)

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read grid rows
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    players = []
    # Find the starting positions of the two players
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'P':
                players.append((r, c))

    start_r1, start_c1 = players[0]
    start_r2, start_c2 = players[1]

    # Use a 4D list to store the minimum moves (distance) to reach each state.
    # Initialize with -1 to indicate unvisited states.
    # State is defined by (player1_row, player1_col, player2_row, player2_col)
    # dist[r1][c1][r2][c2] stores the minimum moves to reach player 1 at (r1, c1) and player 2 at (r2, c2).
    dist = [[[[-1 for _ in range(N)] for _ in range(N)] for _ in range(N)] for _ in range(N)]

    # Queue for BFS
    q = collections.deque()

    # Initial state
    initial_state = (start_r1, start_c1, start_r2, start_c2)
    dist[start_r1][start_c1][start_r2][start_c2] = 0
    q.append(initial_state)

    # Possible directions: UP, DOWN, LEFT, RIGHT
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Perform BFS
    while q:
        r1, c1, r2, c2 = q.popleft()
        d = dist[r1][c1][r2][c2]

        # Check if players have met at the same cell
        if r1 == r2 and c1 == c2:
            print(d)
            return # Found the minimum number of moves

        # Explore reachable states by trying each direction
        for dr, dc in dirs:
            # Calculate the next position for each player independently
            nr1, nc1 = get_next_pos(r1, c1, dr, dc, grid, N)
            nr2, nc2 = get_next_pos(r2, c2, dr, dc, grid, N)

            # The new state is (nr1, nc1, nr2, nc2)
            # Check if this new state has been visited
            if dist[nr1][nc1][nr2][nc2] == -1:
                # Mark as visited and record the distance
                dist[nr1][nc1][nr2][nc2] = d + 1
                # Add the new state to the queue
                q.append((nr1, nc1, nr2, nc2))

    # If the queue becomes empty and the target state was not reached,
    # it's impossible to bring the players together.
    print(-1)

solve()