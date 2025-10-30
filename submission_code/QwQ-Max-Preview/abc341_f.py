import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    W = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    # Adjust to 1-based index
    W = [0] + W
    A = [0] + A
    vertices = list(range(1, N+1))
    # Sort vertices in increasing order of W
    vertices.sort(key=lambda x: W[x])
    
    f = [0] * (N + 1)
    
    for x in vertices:
        wx = W[x]
        adj = edges[x]
        items = []
        for y in adj:
            if W[y] < wx:
                items.append((W[y], f[y] + 1))
        capacity = wx - 1
        if capacity < 0:
            max_val = 0
        else:
            dp = [-1] * (capacity + 1)
            dp[0] = 0
            for w_y, val in items:
                for w in range(capacity, w_y - 1, -1):
                    if dp[w - w_y] != -1:
                        if dp[w] < dp[w - w_y] + val:
                            dp[w] = dp[w - w_y] + val
            max_val = max(dp) if len(dp) > 0 else 0
        f[x] = 1 + max_val
    
    ans = 0
    for i in range(1, N+1):
        ans += A[i] * f[i]
    print(ans)

if __name__ == '__main__':
    main()