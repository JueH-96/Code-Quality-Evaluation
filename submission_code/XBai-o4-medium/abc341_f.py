def main():
    import sys
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
    
    sorted_vertices = sorted(range(N), key=lambda x: W[x])
    c = [0] * N
    
    for x in sorted_vertices:
        items = []
        wx = W[x]
        for neighbor in adj[x]:
            if W[neighbor] < wx:
                items.append((W[neighbor], c[neighbor]))
        capacity = wx - 1
        if not items:
            c[x] = 1
            continue
        dp = [0] * (capacity + 1)
        for w, v in items:
            for cap in range(capacity, w - 1, -1):
                if dp[cap - w] + v > dp[cap]:
                    dp[cap] = dp[cap - w] + v
        max_val = max(dp)
        c[x] = 1 + max_val
    
    total = sum(A[i] * c[i] for i in range(N))
    print(total)

if __name__ == "__main__":
    main()