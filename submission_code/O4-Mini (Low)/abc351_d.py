import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # Flatten index
    def idx(i, j):
        return i * W + j
    
    # Direction vectors
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    
    # Mark non-magnet cells, and whether they are "clean" (no adjacent magnet)
    is_clean = [False] * (H * W)
    non_magnet = [False] * (H * W)
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                non_magnet[idx(i,j)] = True
    # Determine clean
    for i in range(H):
        for j in range(W):
            if not non_magnet[idx(i,j)]:
                continue
            # check adjacent for magnet
            clean = True
            for di, dj in dirs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '#':
                        clean = False
                        break
            is_clean[idx(i,j)] = clean
    
    # DSU over clean cells
    parent = list(range(H*W))
    size = [1]*(H*W)
    def find(x):
        while parent[x]!=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def unite(a,b):
        a = find(a)
        b = find(b)
        if a==b: return
        if size[a]<size[b]:
            a,b = b,a
        parent[b] = a
        size[a] += size[b]
    
    # Union adjacent cleans
    for i in range(H):
        for j in range(W):
            u = idx(i,j)
            if not is_clean[u]: continue
            for di, dj in dirs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = idx(ni,nj)
                    if is_clean[v]:
                        unite(u,v)
    
    # Count component sizes: compress roots
    comp_size = {}
    for i in range(H):
        for j in range(W):
            u = idx(i,j)
            if is_clean[u]:
                r = find(u)
                # we will use DSU size[], but we also record which roots exist
                comp_size[r] = size[find(r)]
    
    # For each component root, count unique adjacent dirty cells
    dirty_adj = {r:0 for r in comp_size}
    # For each dirty cell, see which neighboring comps it touches
    for i in range(H):
        for j in range(W):
            u = idx(i,j)
            if not non_magnet[u] or is_clean[u]:
                continue
            # dirty cell
            neigh_roots = set()
            for di, dj in dirs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = idx(ni,nj)
                    if is_clean[v]:
                        neigh_roots.add(find(v))
            for r in neigh_roots:
                dirty_adj[r] += 1
    
    ans = 1  # at least 1, from any dirty or clean single cell
    # For each clean component, degree = comp_size + dirty_adj
    for r, sz in comp_size.items():
        total = sz + dirty_adj[r]
        if total > ans:
            ans = total
    
    print(ans)

if __name__ == "__main__":
    main()