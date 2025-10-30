# YOUR CODE HERE
from collections import deque

def bfs(start, grid, N):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * N for _ in range(N)]
    queue = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True
    reachable = set()
    
    while queue:
        x, y, dist = queue.popleft()
        reachable.add((x, y, dist))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not visited[nx][ny] and grid[nx][ny] != '#':
                visited[nx][ny] = True
                queue.append((nx, ny, dist + 1))
    
    return reachable

def min_moves_to_meet(N, grid):
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    if len(players) != 2:
        return -1
    
    p1, p2 = players
    reachable_from_p1 = bfs(p1, grid, N)
    reachable_from_p2 = bfs(p2, grid, N)
    
    min_moves = float('inf')
    for x1, y1, dist1 in reachable_from_p1:
        for x2, y2, dist2 in reachable_from_p2:
            if x1 == x2 and y1 == y2:
                min_moves = min(min_moves, max(dist1, dist2))
    
    return min_moves if min_moves != float('inf') else -1

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    grid = data[1:]
    
    result = min_moves_to_meet(N, grid)
    print(result)