def find_min_cost(T, bags):
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # Cost to build empty string is 0

    for i in range(1, len(T) + 1):
        for bag in bags:
            for string in bag:
                if T.startswith(string, i - len(string)) and dp[i - len(string)] != float('inf'):
                    dp[i] = min(dp[i], dp[i - len(string)] + 1)

    return dp[len(T)] if dp[len(T)] != float('inf') else -1

# Read input
T = input().strip()
N = int(input().strip())
bags = [input().strip().split()[1:] for _ in range(N)]

# Solve the problem
min_cost = find_min_cost(T, bags)

# Write output
print(min_cost)