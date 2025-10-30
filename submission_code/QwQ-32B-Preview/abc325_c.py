import sys

class UF:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
        self.num_sets = 0  # to be initialized later

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
        self.num_sets -= 1

def main():
    data = sys.stdin.read().splitlines()
    H_W = data[0].split()
    H = int(H_W[0])
    W = int(H_W[1])
    grid = data[1:H+1]
    
    # Count the number of sensors
    count = 0
    for row in grid:
        count += row.count('#')
    
    uf = UF(H * W)
    uf.num_sets = count
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        x = i * W + j
                        y = ni * W + nj
                        uf.union(x, y)
    
    print(uf.num_sets)

if __name__ == "__main__":
    main()