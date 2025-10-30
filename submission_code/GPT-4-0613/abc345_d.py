from itertools import permutations

def solve():
    N, H, W = map(int, input().split())
    tiles = [sorted(map(int, input().split())) for _ in range(N)]
    for perm in permutations(tiles):
        dp = [[0]*(W+1) for _ in range(H+1)]
        dp[0][0] = 1
        for a, b in perm:
            new_dp = [row[:] for row in dp]
            for i in range(H+1):
                for j in range(W+1):
                    if dp[i][j]:
                        if i+a <= H and j+b <= W:
                            new_dp[i+a][j+b] = 1
                        if i+b <= H and j+a <= W:
                            new_dp[i+b][j+a] = 1
            dp = new_dp
        if dp[H][W]:
            print('Yes')
            return
    print('No')

solve()