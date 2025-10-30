import collections

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]

    player1_r, player1_c = -1, -1
    player2_r, player2_c = -1, -1
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                if player1_r == -1:
                    player1_r, player1_c = r, c
                else:
                    player2_r, player2_c = r, c

    q = collections.deque([(player1_r, player1_c, player2_r, player2_c, 0)])
    visited = set()
    visited.add((player1_r, player1_c, player2_r, player2_c))

    while q:
        r1, c1, r2, c2, dist = q.popleft()
        if r1 == r2 and c1 == c2:
            print(dist)
            return

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in moves:
            nr1, nc1 = r1 + dr, c1 + dc
            nr2, nc2 = r2 + dr, c2 + dc

            valid1 = (0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#')
            valid2 = (0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#')

            next_r1, next_c1 = r1, c1
            next_r2, next_c2 = r2, c2
            if valid1:
                next_r1, next_c1 = nr1, nc1
            if valid2:
                next_r2, next_c2 = nr2, nc2

            if (next_r1, next_c1, next_r2, next_c2) not in visited:
                visited.add((next_r1, next_c1, next_r2, next_c2))
                q.append((next_r1, next_c1, next_r2, next_c2, dist + 1))

    print(-1)

solve()