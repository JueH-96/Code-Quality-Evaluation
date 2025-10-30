def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[idx]) - 1  # convert to 0-based
        idx += 1
        v = int(input[idx]) - 1
        idx += 1
        w = int(input[idx])
        idx += 1
        adj[u].append((v, w))
    
    full_mask = (1 << N) - 1
    INF = 1 << 60
    
    # Initialize DP table
    dp = [[INF] * (1 << N) for _ in range(N)]
    for u in range(N):
        dp[u][1 << u] = 0
    
    # Preprocess masks by bit count
    masks_by_bit_count = [[] for _ in range(N + 1)]
    for mask in range(1, 1 << N):
        cnt = bin(mask).count('1')
        if cnt <= N:
            masks_by_bit_count[cnt].append(mask)
    
    # Iterate over each bit count from 1 to N
    for bit_count in range(1, N + 1):
        for mask in masks_by_bit_count[bit_count]:
            tmp = mask
            while tmp:
                u = (tmp & -tmp).bit_length() - 1
                current_cost = dp[u][mask]
                if current_cost == INF:
                    tmp &= (tmp - 1)
                    continue
                # Explore all edges from u
                for (v, w) in adj[u]:
                    new_mask = mask | (1 << v)
                    if dp[v][new_mask] > current_cost + w:
                        dp[v][new_mask] = current_cost + w
                tmp &= (tmp - 1)  # clear the least significant bit
    
    min_total = min(dp[v][full_mask] for v in range(N))
    if min_total == INF:
        print("No")
    else:
        print(min_total)

if __name__ == "__main__":
    main()