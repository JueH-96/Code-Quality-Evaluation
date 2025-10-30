import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        adj[u].append((v, b, c))

    def check(lam):
        INF = float('-inf')
        dp = [INF] * (N + 1)
        dp[1] = 0.0
        for u in range(1, N + 1):
            current = dp[u]
            if current == INF:
                continue
            for (v, b, c) in adj[u]:
                new_val = current + b - lam * c
                if new_val > dp[v]:
                    dp[v] = new_val
        return dp[N] >= -1e-8

    low = 0.0
    high = 10000.0
    for _ in range(100):
        mid = (low + high) / 2
        if check(mid):
            low = mid
        else:
            high = mid

    print("{0:.15f}".format(low))

if __name__ == '__main__':
    main()