def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    
    A = []
    index = 3
    for _ in range(H):
        A.append([int(data[index + j]) for j in range(W)])
        index += W
    
    from collections import deque
    
    def bfs(sea_level):
        visited = [[False] * W for _ in range(H)]
        queue = deque()
        
        # Start from the perimeter
        for i in range(H):
            if A[i][0] <= sea_level and not visited[i][0]:
                queue.append((i, 0))
                visited[i][0] = True
            if A[i][W-1] <= sea_level and not visited[i][W-1]:
                queue.append((i, W-1))
                visited[i][W-1] = True
        
        for j in range(W):
            if A[0][j] <= sea_level and not visited[0][j]:
                queue.append((0, j))
                visited[0][j] = True
            if A[H-1][j] <= sea_level and not visited[H-1][j]:
                queue.append((H-1, j))
                visited[H-1][j] = True
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and A[nx][ny] <= sea_level:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        
        # Count remaining sections
        remaining = 0
        for i in range(H):
            for j in range(W):
                if not visited[i][j]:
                    remaining += 1
        return remaining
    
    results = []
    for year in range(1, Y + 1):
        sea_level = year
        remaining_area = bfs(sea_level)
        results.append(remaining_area)
    
    for result in results:
        print(result)