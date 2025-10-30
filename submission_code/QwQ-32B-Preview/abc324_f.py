import sys
import math

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))

    low = 0.0
    high = 1e5  # Upper bound for binary search
    tol = 1e-9

    def check(R):
        sum_v = [-float('inf')] * (N + 1)
        sum_v[1] = 0.0
        for u in range(1, N + 1):
            for v, b, c in adj[u]:
                sum_v[v] = max(sum_v[v], sum_v[u] + b - R * c)
        return sum_v[N] >= 0

    while high - low > tol:
        mid = (low + high) / 2
        if check(mid):
            low = mid
        else:
            high = mid

    print("{0:.12f}".format(low))

if __name__ == "__main__":
    main()