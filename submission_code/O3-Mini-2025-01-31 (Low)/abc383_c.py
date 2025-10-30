def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    H, W, D = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # multi-source BFS, starting from each humidifier cell
    q = deque()
    # distance matrix, set to a high value/infinity (or use None)
    dist = [[-1] * W for _ in range(H)]
    
    # initialize the queue with humidifier cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
    
    # four directions: up, down, left, right
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while q:
        i, j = q.popleft()
        # if already used maximum allowed moves, skip further expansion
        if dist[i][j] == D:
            continue
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                # skip if wall
                if grid[ni][nj] == '#':
                    continue
                # if already visited with a distance <= current + 1, skip
                if dist[ni][nj] != -1:
                    continue
                # mark and push
                dist[ni][nj] = dist[i][j] + 1
                q.append((ni, nj))
    
    # Count humidified floor cells that are reached by humidifiers.
    # A humidified cell is a floor '.' or a humidifier cell 'H' that was reached.
    count = 0
    for i in range(H):
        for j in range(W):
            if dist[i][j] != -1:
                if grid[i][j] == '.' or grid[i][j] == 'H':
                    count += 1
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()