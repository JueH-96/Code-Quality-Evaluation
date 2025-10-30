from math import inf, ceil

N = int(input())
D = list(map(int, input().split()))
L1, C1, K1 = map(int, input().split())
L2, C2, K2 = map(int, input().split())

# For each section, precompute all valid (x1, x2, cost) combinations
section_options = []
possible = True

for i in range(N):
    options = []
    d = D[i]
    
    # Try all values of x1 from 0 to K1
    for x1 in range(K1 + 1):
        coverage_from_x1 = x1 * L1
        if coverage_from_x1 >= d:
            # We can use only type-1 sensors
            options.append((x1, 0, x1 * C1))
        else:
            # We need some type-2 sensors as well
            remaining = d - coverage_from_x1
            min_x2 = ceil(remaining / L2)
            if min_x2 <= K2:
                options.append((x1, min_x2, x1 * C1 + min_x2 * C2))
    
    if not options:
        possible = False
        break
    
    section_options.append(options)

if not possible:
    print(-1)
else:
    # DP: dp[i][used1][used2] = min cost to cover first i sections using used1 type-1 and used2 type-2 sensors
    dp = [[[inf for _ in range(K2 + 1)] for _ in range(K1 + 1)] for _ in range(N + 1)]
    dp[0][0][0] = 0
    
    for i in range(N):
        for used1 in range(K1 + 1):
            for used2 in range(K2 + 1):
                if dp[i][used1][used2] == inf:
                    continue
                
                # Try all options for section i
                for x1, x2, cost in section_options[i]:
                    new_used1 = used1 + x1
                    new_used2 = used2 + x2
                    if new_used1 <= K1 and new_used2 <= K2:
                        new_cost = dp[i][used1][used2] + cost
                        dp[i + 1][new_used1][new_used2] = min(dp[i + 1][new_used1][new_used2], new_cost)
    
    # Find the minimum cost among all valid final states
    result = inf
    for used1 in range(K1 + 1):
        for used2 in range(K2 + 1):
            result = min(result, dp[N][used1][used2])
    
    print(result if result != inf else -1)