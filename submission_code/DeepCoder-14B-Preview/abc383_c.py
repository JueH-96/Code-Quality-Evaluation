from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    D = int(input[idx]); idx +=1
    
    grid = []
    for _ in range(H):
        line = input[idx]
        grid.append(list(line))
        idx +=1
    
    # Initialize distance matrix
    INF = H * W + 1
    dist = [ [INF] * W for _ in range(H) ]
    queue = deque()
    
    # Enqueue all H cells and set their distance to 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                queue.append( (i, j) )
    
    # Define directions: up, down, left, right
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    # Perform BFS
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    continue  # Skip walls
                if dist[ni][nj] > dist[i][j] + 1:
                    dist[ni][nj] = dist[i][j] + 1
                    queue.append( (ni, nj) )
    
    # Count all non-wall cells with distance <= D
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                continue
            if dist[i][j] <= D:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()