from heapq import heappush, heappop
from collections import deque

def solve(H, W, grid, N, medicines):
    # Convert grid to 0-based indexing
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
            grid[i][j] = 0 if grid[i][j] == '.' else -1

    # BFS to find distances from start and goal
    def bfs(start):
        queue = deque([start])
        dist = [[-1] * W for _ in range(H)]
        dist[start[0]][start[1]] = 0
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] >= 0 and dist[nx][ny] == -1:
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx, ny))
        return dist

    dist_from_start = bfs(start)
    dist_from_goal = bfs(goal)

    # Add medicines to a priority queue
    pq = []
    for r, c, e in medicines:
        r -= 1
        c -= 1
        if dist_from_start[r][c] != -1 and dist_from_goal[r][c] != -1:
            heappush(pq, (dist_from_start[r][c] + dist_from_goal[r][c], e))

    # Dijkstra's algorithm
    visited = set()
    while pq:
        dist, energy = heappop(pq)
        if (dist, energy) in visited:
            continue
        visited.add((dist, energy))
        if dist <= energy:
            return 'Yes'
        for _ in range(len(pq)):
            next_dist, next_energy = heappop(pq)
            if next_dist <= energy:
                heappush(pq, (next_dist, energy))
            else:
                heappush(pq, (next_dist, next_energy))
                break

    return 'No'

# Read input
H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]
N = int(input())
medicines = [tuple(map(int, input().split())) for _ in range(N)]

# Solve and print result
print(solve(H, W, grid, N, medicines))