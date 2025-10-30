import sys
from collections import deque

def bfs(matrix, visited, h, w, sea_level):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque()
    
    for i in range(h):
        for j in [0, w-1]:
            if matrix[i][j] <= sea_level and not visited[i][j]:
                queue.append((i, j))
                visited[i][j] = True
        for j in range(w):
            if i == 0 or i == h-1:
                if matrix[i][j] <= sea_level and not visited[i][j]:
                    queue.append((i, j))
                    visited[i][j] = True
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and matrix[nx][ny] <= sea_level:
                visited[nx][ny] = True
                queue.append((nx, ny))

def main():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    
    matrix = []
    index = 3
    for _ in range(H):
        row = [int(data[index + i]) for i in range(W)]
        matrix.append(row)
        index += W
    
    results = []
    visited = [[False] * W for _ in range(H)]
    
    for year in range(1, Y + 1):
        bfs(matrix, visited, H, W, year)
        area = sum(1 for i in range(H) for j in range(W) if not visited[i][j])
        results.append(area)
    
    for area in results:
        print(area)

if __name__ == "__main__":
    main()