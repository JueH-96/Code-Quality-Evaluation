# YOUR CODE HERE
from collections import deque

def read_input():
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    return H, W, Y, A

def sink_island(H, W, A, sea_level):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = [[False] * W for _ in range(H)]
    queue = deque()

    # Initialize the queue with border cells
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                if A[i][j] <= sea_level:
                    queue.append((i, j))
                    visited[i][j] = True

    # BFS to sink connected cells
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and A[nx][ny] <= sea_level:
                queue.append((nx, ny))
                visited[nx][ny] = True

    # Count remaining land
    remaining_land = sum(1 for row in visited for cell in row if not cell)
    return remaining_land

def solve():
    H, W, Y, A = read_input()
    for year in range(1, Y + 1):
        remaining_land = sink_island(H, W, A, year)
        print(remaining_land)

solve()