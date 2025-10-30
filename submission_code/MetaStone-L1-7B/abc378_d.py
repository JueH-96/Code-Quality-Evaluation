from collections import deque

def main():
    import sys
    H, W, K = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    starting_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                starting_cells.append((i, j))
    
    count = 0
    for (i, j) in starting_cells:
        mask = 1 << (i * W + j)
        queue = deque()
        queue.append((i, j, 0, mask))
        while queue:
            current_i, current_j, steps, current_mask = queue.popleft()
            if steps == K:
                count += 1
                continue
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni = current_i + di
                nj = current_j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '.' and not (current_mask & (1 << (ni * W + nj))):
                        new_mask = current_mask | (1 << (ni * W + nj))
                        queue.append((ni, nj, steps + 1, new_mask))
    
    print(count)

if __name__ == '__main__':
    main()