import collections

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W, D_val = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    dist = [[-1] * W for _ in range(H)]
    q = collections.deque()
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
                
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while q:
        i, j = q.popleft()
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] != '#' and dist[ni][nj] == -1:
                    new_dist = dist[i][j] + 1
                    dist[ni][nj] = new_dist
                    if new_dist < D_val:
                        q.append((ni, nj))
                        
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1 and dist[i][j] <= D_val:
                count += 1
                
    print(count)

if __name__ == '__main__':
    main()