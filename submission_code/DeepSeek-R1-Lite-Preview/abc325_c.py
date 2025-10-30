def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H_W = sys.stdin.readline().split()
    while len(H_W) < 2:
        H_W += sys.stdin.readline().split()
    H, W = map(int, H_W)
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        while len(line) < W:
            line += sys.stdin.readline().strip()
        grid.append(line)
    
    parent = [i for i in range(H * W)]
    rank = [1] * (H * W)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        xroot = find(x)
        yroot = find(y)
        if xroot == yroot:
            return
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        else:
            parent[yroot] = xroot
            if rank[xroot] == rank[yroot]:
                rank[xroot] += 1
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                current = i * W + j
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        neighbor = ni * W + nj
                        union(current, neighbor)
    
    sensors = [i * W + j for i in range(H) for j in range(W) if grid[i][j] == '#']
    if not sensors:
        print(0)
        return
    roots = set(find(sensor) for sensor in sensors)
    print(len(roots))

if __name__ == "__main__":
    main()