import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges.append(((u, v), (b, c)))
    
    def check_ratio(ratio):
        dp = [-float('inf')] * (n + 1)
        dp[1] = 0
        for u_node in range(1, n + 1):
            if dp[u_node] == -float('inf'):
                continue
            for (u, v), (b, c) in edges:
                if u == u_node:
                    value = b - ratio * c
                    dp[v] = max(dp[v], dp[u] + value)
        return dp[n] >= 0
        
    low = 0
    high = 10000.0 # A large enough upper bound for the ratio
    
    for _ in range(100): # 100 iterations should be enough for precision
        mid = (low + high) / 2
        if check_ratio(mid):
            low = mid
        else:
            high = mid
            
    print(f"{low:.12f}")

if __name__ == '__main__':
    solve()