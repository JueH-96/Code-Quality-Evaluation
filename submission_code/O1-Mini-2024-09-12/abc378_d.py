# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    sys.setrecursionlimit(1000000)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    dp = defaultdict(int)
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                index = i * W + j
                mask = 1 << index
                dp[(i, j, mask)] = 1
    
    for _ in range(K):
        new_dp = defaultdict(int)
        for (i, j, mask), cnt in dp.items():
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                    idx = ni * W + nj
                    if not (mask & (1 << idx)):
                        new_mask = mask | (1 << idx)
                        new_dp[(ni, nj, new_mask)] += cnt
        dp = new_dp
    
    total = sum(dp.values())
    print(total)

if __name__ == "__main__":
    main()