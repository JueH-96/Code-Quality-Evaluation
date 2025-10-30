# YOUR CODE HERE
import collections
import sys

def solve():
    N = int(sys.stdin.readline())
    grid_str = [sys.stdin.readline().strip() for _ in range(N)]

    grid = []
    player_positions = []
    for r in range(N):
        row = []
        for c in range(N):
            char = grid_str[r][c]
            row.append(char)
            if char == 'P':
                player_positions.append((r, c))
        grid.append(row)

    # Players are distinguishable. Their initial positions determine their identities (Player 1 and Player 2).
    # The first 'P' found is Player 1, the second is Player 2.
    p1_start_r, p1_start_c = player_positions[0]
    p2_start_r, p2_start_c = player_positions[1]

    # BFS Queue: stores (r1, c1, r2, c2, moves)
    q = collections.deque()
    q.append((p1_start_r, p1_start_c, p2_start_r, p2_start_c, 0))

    # Visited set: stores (r1, c1, r2, c2) tuples to avoid re-exploring states
    visited = set()
    visited.add((p1_start_r, p1_start_c, p2_start_r, p2_start_c))

    # Directions: (dr, dc) for up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r1, c1, r2, c2, moves = q.popleft()

        # Check if players are on the same cell
        if r1 == r2 and c1 == c2:
            sys.stdout.write(str(moves) + "
")
            return

        for dr, dc in directions:
            # Calculate next *attempted* positions for both players
            next_r1, next_c1 = r1 + dr, c1 + dc
            next_r2, next_c2 = r2 + dr, c2 + dc

            # Determine Player 1's actual new position based on rules
            actual_r1, actual_c1 = r1, c1  # Player 1 stays at current position by default
            # Check bounds and if destination is not an obstacle
            if 0 <= next_r1 < N and 0 <= next_c1 < N and grid[next_r1][next_c1] != '#':
                actual_r1, actual_c1 = next_r1, next_c1

            # Determine Player 2's actual new position based on rules
            actual_r2, actual_c2 = r2, c2  # Player 2 stays at current position by default
            # Check bounds and if destination is not an obstacle
            if 0 <= next_r2 < N and 0 <= next_c2 < N and grid[next_r2][next_c2] != '#':
                actual_r2, actual_c2 = next_r2, next_c2
            
            new_state = (actual_r1, actual_c1, actual_r2, actual_c2)

            # If this new state has not been visited, add it to the queue and visited set
            if new_state not in visited:
                visited.add(new_state)
                q.append((actual_r1, actual_c1, actual_r2, actual_c2, moves + 1))

    # If the queue becomes empty and the target state (players on same cell) was not reached
    sys.stdout.write("-1
")

solve()