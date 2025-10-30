def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    grid = []
    for i in range(1, 1 + n):
        grid.append(list(data[i].strip()))
    
    if n == 1:
        print(''.join(grid[0]))
        return
    
    ring = []
    for j in range(n):
        ring.append(grid[0][j])
    
    for i in range(1, n - 1):
        ring.append(grid[i][n - 1])
    
    for j in range(n - 1, -1, -1):
        ring.append(grid[n - 1][j])
    
    for i in range(n - 2, 0, -1):
        ring.append(grid[i][0])
    
    if ring:
        shifted_ring = [ring[-1]] + ring[:-1]
    else:
        shifted_ring = []
    
    idx = 0
    for j in range(n):
        grid[0][j] = shifted_ring[idx]
        idx += 1
    
    for i in range(1, n - 1):
        grid[i][n - 1] = shifted_ring[idx]
        idx += 1
    
    for j in range(n - 1, -1, -1):
        grid[n - 1][j] = shifted_ring[idx]
        idx += 1
    
    for i in range(n - 2, 0, -1):
        grid[i][0] = shifted_ring[idx]
        idx += 1
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()