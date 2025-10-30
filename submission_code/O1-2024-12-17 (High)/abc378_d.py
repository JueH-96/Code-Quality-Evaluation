def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.read().strip().split()
    H, W, K = map(int, data[:3])
    grid = data[3:]
    
    # Collect all empty cells and map them to indices
    cells = []
    cell_index = {}
    idx = 0
    for r in range(H):
        row = grid[r]
        for c in range(W):
            if row[c] == '.':
                cell_index[(r, c)] = idx
                cells.append((r, c))
                idx += 1
    N = len(cells)
    
    # Build adjacency list
    adjacency = [[] for _ in range(N)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(N):
        r, c = cells[i]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] == '.':
                    j = cell_index[(nr, nc)]
                    adjacency[i].append(j)
    
    visited = [False] * N
    
    def dfs(cur, depth):
        # If we have used K moves, we have a valid path
        if depth == K:
            return 1
        count = 0
        for nxt in adjacency[cur]:
            if not visited[nxt]:
                visited[nxt] = True
                count += dfs(nxt, depth + 1)
                visited[nxt] = False
        return count
    
    total_count = 0
    # Try starting from each empty cell
    for i in range(N):
        visited[i] = True
        total_count += dfs(i, 0)
        visited[i] = False
    
    print(total_count)

# Do not forget to call main()!
main()