def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    grid = []
    for i in range(1, 1+N):
        grid.append(data[i].strip())
    
    ring = []
    for j in range(N):
        ring.append(grid[0][j])
    
    for i in range(1, N-1):
        ring.append(grid[i][N-1])
    
    for j in range(N-1, -1, -1):
        ring.append(grid[N-1][j])
    
    for i in range(N-2, 0, -1):
        ring.append(grid[i][0])
    
    if ring:
        new_ring = [ring[-1]] + ring[:-1]
    else:
        new_ring = []
    
    res = [list(row) for row in grid]
    
    index = 0
    for j in range(N):
        res[0][j] = new_ring[index]
        index += 1
        
    for i in range(1, N-1):
        res[i][N-1] = new_ring[index]
        index += 1
        
    for j in range(N-1, -1, -1):
        res[N-1][j] = new_ring[index]
        index += 1
        
    for i in range(N-2, 0, -1):
        res[i][0] = new_ring[index]
        index += 1
        
    for i in range(N):
        print(''.join(res[i]))
        
if __name__ == "__main__":
    main()