import collections

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    H, W, D = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    dist = [[-1] * W for _ in range(H)]
    q = collections.deque()
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
                
    while q:
        x, y = q.popleft()
        d = dist[x][y]
        if d == D:
            continue
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] == '#' or dist[nx][ny] != -1:
                    continue
                nd = d + 1
                dist[nx][ny] = nd
                if nd < D:
                    q.append((nx, ny))
                    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1:
                count += 1
                
    print(count)

if __name__ == "__main__":
    main()