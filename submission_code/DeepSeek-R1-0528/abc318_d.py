import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    graph = [[0] * n for _ in range(n)]
    index = 1
    for i in range(n-1):
        for j in range(i+1, n):
            d_val = int(data[index])
            index += 1
            graph[i][j] = d_val
            graph[j][i] = d_val

    total_masks = 1 << n
    dp = [-10**18] * total_masks
    dp[0] = 0
    
    for mask in range(total_masks):
        if dp[mask] == -10**18:
            continue
        available = []
        for i in range(n):
            if not (mask & (1 << i)):
                available.append(i)
        m = len(available)
        for i in range(m):
            for j in range(i+1, m):
                u = available[i]
                v = available[j]
                new_mask = mask | (1 << u) | (1 << v)
                total_weight = dp[mask] + graph[u][v]
                if total_weight > dp[new_mask]:
                    dp[new_mask] = total_weight
                    
    print(max(dp))

if __name__ == '__main__':
    main()