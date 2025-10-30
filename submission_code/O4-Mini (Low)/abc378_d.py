import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline

    H, W, K = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]

    # Map each cell (i,j) to an index 0..H*W-1
    # Precompute which indices are empty, and for each idx its list of neighbor idxs that are empty.
    empty = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                empty.append(i * W + j)

    # Precompute adjacency for empty cells
    neighbors = [[] for _ in range(H * W)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                idx = i * W + j
                for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                        neighbors[idx].append(ni * W + nj)

    # DFS with bitmask visited
    from functools import lru_cache

    # We'll write a recursive function that counts walks of total length K+1.
    # depth = number of positions visited so far
    # cur = current cell index
    # visited_mask = bitmask of visited cells
    @lru_cache(maxsize=None)
    def dfs(cur, visited_mask, depth):
        if depth == K + 1:
            return 1
        total = 0
        for nxt in neighbors[cur]:
            bit = 1 << nxt
            if not (visited_mask & bit):
                total += dfs(nxt, visited_mask | bit, depth + 1)
        return total

    ans = 0
    # Start DFS from every empty cell
    for start in empty:
        init_mask = (1 << start)
        ans += dfs(start, init_mask, 1)

    print(ans)


if __name__ == "__main__":
    main()