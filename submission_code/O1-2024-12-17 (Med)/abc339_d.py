def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    N = int(input().strip())
    grid = [input().rstrip("
") for _ in range(N)]
    
    # Find the positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    (r1, c1), (r2, c2) = players[0], players[1]

    # BFS queue and visited set
    queue = deque()
    queue.append((r1, c1, r2, c2, 0))
    visited = set([(r1, c1, r2, c2)])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def can_move(r, c, nr, nc):
        # Check if we can move from (r,c) to (nr,nc)
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != '#':
            return nr, nc
        return r, c

    # Perform BFS
    while queue:
        r1, c1, r2, c2, dist = queue.popleft()
        
        # Check if players meet
        if (r1, c1) == (r2, c2):
            print(dist)
            return
        
        # Try moving in all directions
        for dr, dc in directions:
            nr1, nc1 = can_move(r1, c1, r1 + dr, c1 + dc)
            nr2, nc2 = can_move(r2, c2, r2 + dr, c2 + dc)
            if (nr1, nc1, nr2, nc2) not in visited:
                visited.add((nr1, nc1, nr2, nc2))
                queue.append((nr1, nc1, nr2, nc2, dist + 1))
    
    # Impossible to meet
    print(-1)

# Do not forget to call main()
main()