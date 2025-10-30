from collections import defaultdict

def main():
    H, W, K = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    dp = defaultdict(int)
    
    # Initialize DP with starting cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                idx = i * W + j
                mask = 1 << idx
                dp[(i, j, mask)] += 1
    
    # Perform BFS for K steps
    for _ in range(K):
        next_dp = defaultdict(int)
        for (i, j, mask), cnt in dp.items():
            # Explore all four directions
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '.':
                        new_idx = ni * W + nj
                        if not (mask & (1 << new_idx)):
                            new_mask = mask | (1 << new_idx)
                            next_dp[(ni, nj, new_mask)] += cnt
        dp = next_dp
    
    # Sum all valid paths after K steps
    print(sum(dp.values()))

if __name__ == "__main__":
    main()