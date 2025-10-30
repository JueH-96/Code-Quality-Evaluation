def main():
    import sys
    sys.setrecursionlimit(1000000)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Identify empty cells and map to indices
    empty_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    N = len(empty_cells)
    pos_to_index = {(i, j): idx for idx, (i, j) in enumerate(empty_cells)}
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Precompute neighbors for each empty cell
    neighbors = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(empty_cells):
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                neighbor_idx = pos_to_index[(ni, nj)]
                neighbors[idx].append(neighbor_idx)
    
    # Iterative DFS with stack
    count = 0
    stack = []
    
    # Initialize stack with starting positions
    for start_idx in range(N):
        stack.append((start_idx, 0, 1 << start_idx))
    
    while stack:
        pos, moves, mask = stack.pop()
        if moves == K:
            count += 1
        elif moves < K:
            for neighbor in neighbors[pos]:
                if not (mask & (1 << neighbor)):
                    stack.append((neighbor, moves + 1, mask | (1 << neighbor)))
    
    print(count)

if __name__ == '__main__':
    main()