def main():
    import sys
    from collections import deque
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    grid = data[1:]
    
    # Find the two players' starting positions.
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    # There are exactly two players.
    (x1_start, y1_start), (x2_start, y2_start) = players[0], players[1]
    
    # Precompute for every cell that is not an obstacle, for each move direction,
    # what is the next cell if a player at that cell attempts to move in that direction.
    # The rules: if the neighboring cell exists in grid and is not an obstacle,
    # then the player goes there; otherwise the player remains.
    # Directions order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    next_move = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            # Only precompute for where a player can actually be.
            if grid[i][j] == '#':
                continue
            moves = []
            for dx, dy in directions:
                nx, ny = i + dx, j + dy
                if 0 <= nx < N and 0 <= ny < N and grid[nx][ny] != '#':
                    moves.append((nx, ny))
                else:
                    moves.append((i, j))
            next_move[i][j] = moves

    # We'll do a BFS on the state space of (player1_cell, player2_cell).
    # Our state is a 4-tuple (x1, y1, x2, y2). In each move we choose one of 4 directions
    # and for both players apply the same direction (with the rule that if the adjacent cell is valid,
    # they move; otherwise they remain). We want the minimum number of moves so that the two players
    # come to the same cell.
    #
    # To speed up the lookup we encode a state (x1, y1, x2, y2) as a single integer:
    #   state_id = (((x1 * N + y1) * N + x2) * N + y2)
    #
    def encode(x1, y1, x2, y2):
        return (((x1 * N + y1) * N + x2) * N + y2)
    
    total_states = N**4  # maximum number of states (only empty cells will really be reached)
    # Use a bytearray for visited markers. Each entry is 0 (not visited) or 1 (visited).
    visited = bytearray(total_states)
    
    # Initialize BFS.
    start_state = encode(x1_start, y1_start, x2_start, y2_start)
    dq = deque()
    dq.append((start_state, 0))
    visited[start_state] = 1

    while dq:
        state, dist = dq.popleft()
        # Decode the state back into the two players' positions.
        # Our encoding: state = (((x1 * N + y1) * N + x2) * N + y2)
        s = state
        y2 = s % N
        s //= N
        x2 = s % N
        s //= N
        y1 = s % N
        x1 = s // N

        # Try each of the four moves.
        for d in range(4):
            nx1, ny1 = next_move[x1][y1][d]
            nx2, ny2 = next_move[x2][y2][d]
            # Check if the two players have met.
            if nx1 == nx2 and ny1 == ny2:
                sys.stdout.write(str(dist + 1))
                return
            new_state = (((nx1 * N + ny1) * N + nx2) * N + ny2)
            if visited[new_state] == 0:
                visited[new_state] = 1
                dq.append((new_state, dist + 1))
    sys.stdout.write("-1")
    
if __name__ == '__main__':
    main()