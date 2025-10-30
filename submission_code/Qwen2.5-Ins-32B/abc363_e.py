import sys
from collections import deque

def solve(H, W, Y, A):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    sea_level = 0
    area = H * W
    result = []

    for year in range(1, Y + 1):
        sea_level += 1
        queue = deque()
        
        # Add all boundary cells to the queue
        for i in range(H):
            for j in [0, W-1]:
                if A[i][j] <= sea_level:
                    queue.append((i, j))
                    A[i][j] = -1  # Mark as visited/sunken
        for j in range(W):
            for i in [0, H-1]:
                if A[i][j] <= sea_level:
                    queue.append((i, j))
                    A[i][j] = -1  # Mark as visited/sunken
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and A[nx][ny] <= sea_level:
                    queue.append((nx, ny))
                    A[nx][ny] = -1  # Mark as visited/sunken
                    area -= 1
        
        result.append(area)
    
    return result

if __name__ == "__main__":
    H, W, Y = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    result = solve(H, W, Y, A)
    for area in result:
        print(area)