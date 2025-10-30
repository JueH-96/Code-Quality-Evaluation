import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    max_ratio = 0.0

    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))
        ratio = b / c
        if ratio > max_ratio:
            max_ratio = ratio

    low = 0.0
    high = max_ratio

    # Pre-allocate the dp array
    dp = [-float('inf')] * (N + 1)

    for _ in range(100):
        mid = (low + high) / 2
        # Reset dp array
        dp[1] = 0.0
        for i in range(2, N + 1):
            dp[i] = -float('inf')

        for u in range(1, N + 1):
            if dp[u] == -float('inf'):
                continue
            for (v, b, c) in adj[u]:
                new_val = dp[u] + (b - mid * c)
                if new_val > dp[v]:
                    dp[v] = new_val

        if dp[N] >= 0:
            low = mid
        else:
            high = mid

    print("{0:.15f}".format(low))

if __name__ == "__main__":
    main()