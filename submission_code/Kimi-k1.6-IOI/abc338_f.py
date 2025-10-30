def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(N + 1)]  # 1-based indexing
    
    for _ in range(M):
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        w = int(data[idx])
        idx += 1
        edges[u].append((v, w))
    
    INF = float('inf')
    size = 1 << N
    dp = [[INF] * (N + 1) for _ in range(size)]
    
    for u in range(1, N + 1):
        mask = 1 << (u - 1)
        dp[mask][u] = 0
    
    masks = []
    for mask in range(1, size):
        masks.append(mask)
    
    masks.sort(key=lambda x: bin(x).count('1'))
    
    for mask in masks:
        for u in range(1, N + 1):
            if not (mask & (1 << (u - 1))):
                continue
            current_cost = dp[mask][u]
            if current_cost == INF:
                continue
            for (v, w) in edges[u]:
                new_mask = mask | (1 << (v - 1))
                if dp[new_mask][v] > current_cost + w:
                    dp[new_mask][v] = current_cost + w
    
    full_mask = (1 << N) - 1
    min_total = INF
    for u in range(1, N + 1):
        if dp[full_mask][u] < min_total:
            min_total = dp[full_mask][u]
    
    if min_total == INF:
        print("No")
    else:
        print(min_total)

if __name__ == '__main__':
    main()