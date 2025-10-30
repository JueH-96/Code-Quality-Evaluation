import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges.append(((u, v), (b, c)))
    
    def get_max_value_sum(ratio):
        dp = {i: -float('inf') for i in range(1, n + 1)}
        dp[1] = 0
        for u_node in range(1, n + 1):
            if dp[u_node] == -float('inf'):
                continue
            for edge_info in edges:
                (u, v), (b, c) = edge_info
                if u == u_node:
                    value = b - ratio * c
                    dp[v] = max(dp[v], dp[u] + value)
        return dp[n]
        
    low = 0.0
    high = 10000.0 # Upper bound, maybe max(b_i/c_i)? Let's use 10000 as a safe upper bound.
    
    for _ in range(100): # 100 iterations for precision
        mid = (low + high) / 2.0
        max_val_sum = get_max_value_sum(mid)
        if max_val_sum >= 0:
            low = mid
        else:
            high = mid
            
    print(f"{low:.12f}")

if __name__ == '__main__':
    solve()