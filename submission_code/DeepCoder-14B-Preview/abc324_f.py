def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    max_ratio = 0.0
    
    for _ in range(M):
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        c = int(data[idx])
        idx += 1
        
        adj[u].append((v, b, c))
        current_ratio = b / c
        if current_ratio > max_ratio:
            max_ratio = current_ratio
    
    left = 0.0
    right = max_ratio if max_ratio > 0 else 1.0  # Handle case where all edges have ratio 0
    
    for _ in range(100):
        mid = (left + right) / 2
        
        max_sum = [-float('inf')] * (N + 1)
        max_sum[1] = 0.0
        
        for u in range(1, N):
            if max_sum[u] == -float('inf'):
                continue
            
            for (v, b, c) in adj[u]:
                new_sum = max_sum[u] + (b - mid * c)
                if new_sum > max_sum[v]:
                    max_sum[v] = new_sum
        
        if max_sum[N] >= 0:
            left = mid
        else:
            right = mid
    
    final_r = (left + right) / 2
    print("{0:.15f}".format(final_r))

if __name__ == "__main__":
    main()