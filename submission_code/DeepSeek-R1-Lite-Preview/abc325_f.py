import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    D = [int(data[i]) for i in range(idx, idx + N)]
    idx += N
    L1 = int(data[idx])
    C1 = int(data[idx + 1])
    K1 = int(data[idx + 2])
    idx += 3
    L2 = int(data[idx])
    C2 = int(data[idx + 1])
    K2 = int(data[idx + 2])
    
    # Precompute possible (x, y, cost) for each section
    sections = []
    for d in D:
        options = []
        max_x = int(math.ceil(d / L1)) if L1 != 0 else 0
        for x in range(0, max_x + 1):
            coverage = x * L1
            if coverage >= d:
                y = 0
            else:
                y = int(math.ceil((d - coverage) / L2))
            cost = x * C1 + y * C2
            options.append((x, y, cost))
        sections.append(options)
    
    # Initialize DP
    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0
    
    for options in sections:
        # To prevent using the same sensors for multiple sections, we iterate backwards
        new_dp = [row[:] for row in dp]
        for k1 in range(K1 + 1):
            for k2 in range(K2 + 1):
                if dp[k1][k2] < INF:
                    for x, y, cost in options:
                        if k1 + x <= K1 and k2 + y <= K2:
                            if new_dp[k1 + x][k2 + y] > dp[k1][k2] + cost:
                                new_dp[k1 + x][k2 + y] = dp[k1][k2] + cost
        dp = new_dp
    
    # Find the minimum cost
    min_cost = INF
    for k1 in range(K1 + 1):
        for k2 in range(K2 + 1):
            if dp[k1][k2] < min_cost:
                min_cost = dp[k1][k2]
    
    if min_cost != INF:
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()