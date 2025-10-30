def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx +=1
    
    edges = [[] for _ in range(N+1)]  # 1-based
    max_ratio = 0.0
    
    for _ in range(M):
        u = int(data[idx])
        idx +=1
        v = int(data[idx])
        idx +=1
        b = int(data[idx])
        idx +=1
        c = int(data[idx])
        idx +=1
        edges[u].append( (v, b, c) )
        ratio = b / c
        if ratio > max_ratio:
            max_ratio = ratio
    
    low = 0.0
    high = max_ratio
    
    # Binary search for 100 iterations
    for _ in range(100):
        mid = (low + high) / 2
        max_sum = [ -float('inf') ] * (N +1)
        max_sum[1] = 0.0
        for u in range(1, N+1):
            if max_sum[u] == -float('inf'):
                continue
            for (v, b, c) in edges[u]:
                new_sum = max_sum[u] + (b - mid * c)
                if new_sum > max_sum[v]:
                    max_sum[v] = new_sum
        if max_sum[N] >= -1e-12:  # Considering floating point errors
            low = mid
        else:
            high = mid
    
    print("{0:.15f}".format(low))
    
if __name__ == "__main__":
    main()