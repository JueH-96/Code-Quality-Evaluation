def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    grid = data[2:2+H]
    
    visited = [[False] * W for _ in range(H)]
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    def bfs(start_i, start_j):
        queue = deque()
        queue.append((start_i, start_j))
        visited[start_i][start_j] = True
        while queue:
            i, j = queue.popleft()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if not visited[ni][nj] and grid[ni][nj] == '#':
                        visited[ni][nj] = True
                        queue.append((ni, nj))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                bfs(i, j)
                count += 1
    
    print(count)

if __name__ == '__main__':
    main()