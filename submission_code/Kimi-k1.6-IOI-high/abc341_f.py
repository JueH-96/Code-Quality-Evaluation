import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[ptr]) - 1
        ptr += 1
        v = int(input[ptr]) - 1
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)
    
    W = list(map(int, input[ptr:ptr+N]))
    ptr += N
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    sorted_indices = sorted(range(N), key=lambda x: W[x])
    m = [1] * N
    
    for idx in sorted_indices:
        w_i = W[idx]
        capacity = w_i - 1
        eligible = []
        for neighbor in adj[idx]:
            if W[neighbor] < w_i:
                eligible.append((W[neighbor], m[neighbor]))
        
        if not eligible:
            continue
        
        max_val = 0
        if capacity >= 0:
            dp = [-float('inf')] * (capacity + 1)
            dp[0] = 0
            for (w_j, val) in eligible:
                for w in range(capacity, w_j - 1, -1):
                    if dp[w - w_j] + val > dp[w]:
                        dp[w] = dp[w - w_j] + val
            max_val = max(dp)
        
        m[idx] = 1 + max_val
    
    ans = sum(a * mi for a, mi in zip(A, m))
    print(ans)

if __name__ == '__main__':
    main()