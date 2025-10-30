from collections import deque

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]

    players = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                players.append((r, c))

    start_p1 = players[0]
    start_p2 = players[1]

    queue = deque([((start_p1[0], start_p1[1]), (start_p2[0], start_p2[1]), 0)])
    visited = set([((start_p1[0], start_p1[1]), (start_p2[0], start_p2[1]))])

    while queue:
        (r1, c1), (r2, c2), moves = queue.popleft()

        if (r1, c1) == (r2, c2):
            print(moves)
            return

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr1, nc1 = r1 + dr, c1 + dc
            nr2, nc2 = r2 + dr, c2 + dc

            can_move1 = 0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#'
            can_move2 = 0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#'

            next_r1, next_c1 = (nr1, nc1) if can_move1 else (r1, c1)
            next_r2, next_c2 = (nr2, nc2) if can_move2 else (r2, c2)

            next_state = ((next_r1, next_c1), (next_r2, next_c2))
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state[0], next_state[1], moves + 1))

    print(-1)

solve()