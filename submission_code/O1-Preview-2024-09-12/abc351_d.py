# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Direction vectors: Up, Down, Left, Right
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# Determine passable cells
passable = [[False]*W for _ in range(H)]
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            has_adjacent_magnet = False
            for k in range(4):
                ni = i + dx[k]
                nj = j + dy[k]
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '#':
                        has_adjacent_magnet = True
                        break
            if not has_adjacent_magnet:
                passable[i][j] = True

# Union-Find (Disjoint Set Union) implementation
class DSU:
    def __init__(self, N):
        self.parent = list(range(N))
        self.size = [1]*N

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def unite(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

    def component_size(self, x):
        return self.size[self.find(x)]

N = H * W
dsu = DSU(N)

# For each passable cell, union with adjacent passable cells
for i in range(H):
    for j in range(W):
        if passable[i][j]:
            current = i * W + j
            for k in range(4):
                ni = i + dx[k]
                nj = j + dy[k]
                if 0 <= ni < H and 0 <= nj < W:
                    if passable[ni][nj]:
                        neighbor = ni * W + nj
                        dsu.unite(current, neighbor)

# Compute degree of freedom for each cell
max_degree = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            if passable[i][j]:
                degree = dsu.component_size(i * W + j)
            else:
                degree = 1
            if degree > max_degree:
                max_degree = degree

print(max_degree)