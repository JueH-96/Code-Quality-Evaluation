X, Y, Z = map(int, input().split())
S = input().strip()
n = len(S)
INF = float('inf')

# Initialize DP table
dp = [[INF] * 2 for _ in range(n + 1)]
dp[0][0] = 0
dp[0][1] = Z  # Press Caps once before processing any characters

for i in range(n):
    for cap in [0, 1]:
        if dp[i][cap] == INF:
            continue
        current_char = S[i]
        # Check Action 1: press 'a' only
        if (cap == 0 and current_char == 'a') or (cap == 1 and current_char == 'A'):
            cost = X
            new_cap_after = cap
            for toggle in [0, 1]:
                new_cap = new_cap_after ^ toggle
                new_cost = dp[i][cap] + cost + toggle * Z
                if new_cost < dp[i+1][new_cap]:
                    dp[i+1][new_cap] = new_cost
        # Check Action 2: press 'a' and Shift
        if (cap == 0 and current_char == 'A') or (cap == 1 and current_char == 'a'):
            cost = Y
            new_cap_after = cap
            for toggle in [0, 1]:
                new_cap = new_cap_after ^ toggle
                new_cost = dp[i][cap] + cost + toggle * Z
                if new_cost < dp[i+1][new_cap]:
                    dp[i+1][new_cap] = new_cost

print(min(dp[n][0], dp[n][1]))