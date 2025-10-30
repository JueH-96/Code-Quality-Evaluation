class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Y = int(data[2])
    A = [int(x) for x in data[3:]]
    
    # Flatten grid to 1D
    grid = []
    for i in range(H):
        for j in range(W):
            grid.append(A[i*W + j])
    
    # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Boundary sections
    boundary = set()
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                boundary.add(i * W + j)
    
    # Sort sections by elevation
    sections = sorted([(grid[i], i) for i in range(H*W)], key=lambda x: x[0])
    
    # Union-Find
    uf = UnionFind(H*W + 1)
    sea = H*W
    
    # Sinking years
    sinking_years = [0] * (H*W)
    
    # Process sections in order of elevation
    for t, idx in sections:
        i = idx // W
        j = idx % W
        if idx in boundary:
            uf.union(idx, sea)
            sinking_years[idx] = t
        else:
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    nidx = ni * W + nj
                    if grid[nidx] <= t:
                        uf.union(idx, nidx)
            # Check if connected to sea
            if uf.find(idx) == uf.find(sea):
                sinking_years[idx] = t
            else:
                sinking_years[idx] = float('inf')
    
    # Sort sinking years
    sinking_years.sort()
    
    # Compute prefix sums
    total_sections = H * W
    prefix_sunk = [0] * (Y + 1)
    j = 0
    for y in range(1, Y + 1):
        while j < total_sections and sinking_years[j] <= y:
            j += 1
        prefix_sunk[y] = j
    
    # Output for each year
    for y in range(1, Y + 1):
        above = total_sections - prefix_sunk[y]
        print(above)

if __name__ == "__main__":
    main()