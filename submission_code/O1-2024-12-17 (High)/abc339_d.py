def main():
    import sys
    from collections import deque

    input = sys.stdin.readline

    N = int(input())
    grid = [input().rstrip('
') for _ in range(N)]

    # Locate the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    (r1, c1), (r2, c2) = players

    # Function to move a player one step in direction (dr, dc) if possible
    def try_move(r, c, dr, dc):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != '#':
            return nr, nc
        else:
            return r, c

    # BFS over states: (row_player1, col_player1, row_player2, col_player2)
    queue = deque()
    visited = set()
    queue.append((r1, c1, r2, c2, 0))
    visited.add((r1, c1, r2, c2))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r1, c1, r2, c2, dist = queue.popleft()

        # Check if both players share the same position
        if (r1, c1) == (r2, c2):
            print(dist)
            return

        # Try moving in each of the four directions
        for dr, dc in directions:
            nr1, nc1 = try_move(r1, c1, dr, dc)
            nr2, nc2 = try_move(r2, c2, dr, dc)
            if (nr1, nc1, nr2, nc2) not in visited:
                visited.add((nr1, nc1, nr2, nc2))
                queue.append((nr1, nc1, nr2, nc2, dist + 1))

    # If no way to bring the players together:
    print(-1)

# Do not forget to call main() at the end!
if __name__ == "__main__":
    main()