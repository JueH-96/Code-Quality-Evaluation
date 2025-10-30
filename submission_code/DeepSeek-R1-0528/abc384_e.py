import heapq

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    P0 = int(data[3])
    Q0 = int(data[4])
    index = 5
    grid = []
    for i in range(H):
        row = list(map(int, data[index:index+W]))
        index += W
        grid.append(row)
    
    start_i = P0 - 1
    start_j = Q0 - 1
    total = grid[start_i][start_j]
    visited = [[False] * W for _ in range(H)]
    visited[start_i][start_j] = True
    pq = []
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in dirs:
        ni = start_i + dx
        nj = start_j + dy
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            heapq.heappush(pq, (grid[ni][nj] * X, ni, nj))
            
    while pq:
        req, i, j = heapq.heappop(pq)
        if visited[i][j]:
            continue
        if total <= req:
            break
        visited[i][j] = True
        total += grid[i][j]
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                heapq.heappush(pq, (grid[ni][nj] * X, ni, nj))
                
    print(total)

if __name__ == "__main__":
    main()