# YOUR CODE HERE
from collections import deque

def solve():
    N = int(input())
    grid = [input() for _ in range(N)]
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    visited = set()
    queue = deque([(players[0], players[1], 0)])
    
    while queue:
        p1, p2, moves = queue.popleft()
        if p1 == p2:
            print(moves)
            return
        for dx, dy in directions:
            np1 = (p1[0] + dx, p1[1] + dy)
            np2 = (p2[0] + dx, p2[1] + dy)
            if 0 <= np1[0] < N and 0 <= np1[1] < N and grid[np1[0]][np1[1]] != '#' and np1 not in visited:
                if 0 <= np2[0] < N and 0 <= np2[1] < N and grid[np2[0]][np2[1]] != '#' and np2 not in visited:
                    visited.add(np1)
                    queue.append((np1, np2, moves + 1))
    print(-1)

solve()