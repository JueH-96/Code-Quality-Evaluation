def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    grid = []
    for _ in range(H):
        grid.append(input[idx].strip())
        idx += 1
    
    grid_passable = [[c == '.' for c in row] for row in grid]
    directions = [ (-1, 0), (1, 0), (0, -1), (0, 1) ]
    answer = 0
    
    def dfs(i, j, steps, mask):
        if steps == K:
            return 1
        total = 0
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid_passable[ni][nj]:
                    idx_n = ni * W + nj
                    if not (mask & (1 << idx_n)):
                        total += dfs(ni, nj, steps + 1, mask | (1 << idx_n))
        return total
    
    for i in range(H):
        for j in range(W):
            if grid_passable[i][j]:
                idx_start = i * W + j
                answer += dfs(i, j, 0, 1 << idx_start)
    
    print(answer)

if __name__ == "__main__":
    main()