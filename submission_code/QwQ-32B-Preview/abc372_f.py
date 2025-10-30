import sys
from collections import deque

def main():
    import sys
    from collections import deque
    MOD = 998244353
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    extra_edges = []
    for _ in range(M):
        X, Y = map(int, sys.stdin.readline().split())
        extra_edges.append((X - 1, Y - 1))  # 0-based indexing

    dp = deque([0] * N)
    dp[0] = 1  # Starting at vertex 1 (0-based)

    for _ in range(K):
        dp.rotate(1)  # Cycle transition
        for X, Y in extra_edges:
            dp[Y] = (dp[Y] + dp[X]) % MOD  # Extra edge transition

    print(sum(dp) % MOD)

if __name__ == "__main__":
    main()