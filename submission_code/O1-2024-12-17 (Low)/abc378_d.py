def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    H, W, K = map(int, input_data[:3])
    grid_chars = input_data[3:]

    # Convert (row, col) to a single index idx = row*W + col.
    # We'll skip blocked cells in adjacency construction.
    # adjacency[idx] = list of neighboring indices that are empty.
    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    blocked = [False] * (H * W)
    for r in range(H):
        for c in range(W):
            if grid_chars[r][c] == '#':
                blocked[r*W + c] = True

    adjacency = [[] for _ in range(H * W)]
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for r in range(H):
        for c in range(W):
            idx = r*W + c
            if blocked[idx]:
                continue
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if in_bounds(nr, nc):
                    neighbor_idx = nr*W + nc
                    if not blocked[neighbor_idx]:
                        adjacency[idx].append(neighbor_idx)

    visited = [False]*(H*W)
    sys.setrecursionlimit(10**7)
    ans = 0

    def dfs(cur, depth):
        nonlocal ans
        if depth == K:
            ans += 1
            return
        for nxt in adjacency[cur]:
            if not visited[nxt]:
                visited[nxt] = True
                dfs(nxt, depth+1)
                visited[nxt] = False

    # Try each empty cell as a start
    for cell_idx in range(H*W):
        if not blocked[cell_idx]:
            visited[cell_idx] = True
            dfs(cell_idx, 0)
            visited[cell_idx] = False

    print(ans)


# Do not forget to call main()!
if __name__ == "__main__":
    main()