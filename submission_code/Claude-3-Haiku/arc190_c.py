import sys
from math import comb

MOD = 998244353

def solve():
    H, W = map(int, input().split())
    A = [[int(x) for x in input().split()] for _ in range(H)]
    Q, sh, sw = map(int, input().split())
    h, w = sh, sw

    for _ in range(Q):
        d, a = input().split()
        a = int(a)
        if d == 'L':
            w = max(w - 1, 1)
        elif d == 'R':
            w = min(w + 1, W)
        elif d == 'U':
            h = max(h - 1, 1)
        else:
            h = min(h + 1, H)
        A[h-1][w-1] = a

        total = 0
        for i in range(H):
            for j in range(W):
                if (i, j) == (0, 0) or (i, j) == (H-1, W-1):
                    continue
                path_count = comb(i + j - 1, i - 1)
                total += path_count * A[i][j] % MOD
        print(total % MOD)

solve()