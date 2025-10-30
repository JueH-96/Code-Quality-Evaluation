# YOUR CODE HERE
import sys
import sys
import sys
from sys import stdin
def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    H_W = sys.stdin.readline().split()
    while len(H_W) < 2:
        H_W += sys.stdin.readline().split()
    H, W = map(int, H_W)
    grid = []
    for _ in range(H):
        row = sys.stdin.readline().strip()
        while len(row) < W:
            row += sys.stdin.readline().strip()
        grid.append(row)
    traversable = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                has_magnet = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            has_magnet = True
                            break
                if not has_magnet:
                    traversable[i][j] = True
    parent = [-1]*(H*W)
    size = [1]*(H*W)
    def find(x):
        while parent[x] != x:
            if parent[x] == -1:
                return -1
            if parent[parent[x]] != parent[x]:
                parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root == -1 or y_root == -1:
            return
        if x_root == y_root:
            return
        if size[x_root] < size[y_root]:
            parent[x_root] = y_root
            size[y_root] += size[x_root]
        else:
            parent[y_root] = x_root
            size[x_root] += size[y_root]
    for i in range(H):
        for j in range(W):
            if traversable[i][j]:
                idx = i * W + j
                parent[idx] = idx
    for i in range(H):
        for j in range(W):
            if traversable[i][j]:
                idx = i * W + j
                for di, dj in [(0,1),(1,0)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and traversable[ni][nj]:
                        nidx = ni * W + nj
                        union(idx, nidx)
    count = {}
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' :
                # Check if non-traversable
                if not traversable[i][j]:
                    adjacent_roots = set()
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W and traversable[ni][nj]:
                            nidx = ni * W + nj
                            nroot = find(nidx)
                            if nroot != -1:
                                adjacent_roots.add(nroot)
                    for root in adjacent_roots:
                        if root in count:
                            count[root] +=1
                        else:
                            count[root] =1
    max_degree = 1
    for i in range(H):
        for j in range(W):
            if traversable[i][j]:
                idx = i * W + j
                root = find(idx)
                if root == -1:
                    continue
                current_size = size[root]
                current_count = count.get(root,0)
                degree = current_size + current_count
                if degree > max_degree:
                    max_degree = degree
    print(max_degree)