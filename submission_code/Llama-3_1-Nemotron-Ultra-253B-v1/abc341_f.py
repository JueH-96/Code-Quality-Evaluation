import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    W = list(map(int, stdin.readline().split()))
    A = list(map(int, stdin.readline().split()))
    
    sorted_vertices = sorted(range(1, N+1), key=lambda x: W[x-1])
    f = [0] * (N + 1)
    
    for x in sorted_vertices:
        w_x = W[x-1]
        capacity = w_x - 1
        items = []
        for y in adj[x]:
            if W[y-1] < w_x:
                items.append((W[y-1], f[y]))
        
        if not items:
            f[x] = 1
            continue
        
        total_weight = sum(w for w, _ in items)
        if total_weight <= capacity:
            max_sum = sum(v for _, v in items)
        else:
            dp = [-1 << 60] * (capacity + 1)
            dp[0] = 0
            for w, v in items:
                for j in range(capacity, w-1, -1):
                    if dp[j - w] + v > dp[j]:
                        dp[j] = dp[j - w] + v
            max_sum = max(dp)
        
        f[x] = 1 + max_sum
    
    total = sum(a * f[i+1] for i, a in enumerate(A))
    print(total)

if __name__ == '__main__':
    main()