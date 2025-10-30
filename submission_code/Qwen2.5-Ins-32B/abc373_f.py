import sys

def max_happiness(N, W, items):
    # Precompute the maximum happiness for each item type up to its maximum possible quantity
    max_happiness_per_type = []
    for w, v in items:
        happiness = [0]
        for k in range(1, 100000000000):  # Since k_i can be up to 10^10, but we only need to check up to v
            h = k * v - k * k
            if h <= happiness[-1]:
                break
            happiness.append(h)
        max_happiness_per_type.append((w, happiness))
    
    # Initialize DP table
    dp = [0] * (W + 1)
    
    # Fill DP table
    for w, happiness in max_happiness_per_type:
        for weight in range(W, w - 1, -1):
            for h in happiness:
                if weight - w >= 0:
                    dp[weight] = max(dp[weight], dp[weight - w] + h)
    
    return dp[W]

# Read input
input = sys.stdin.read
data = input().split()
N, W = map(int, [data[0], data[1]])
items = [(int(data[i]), int(data[i + 1])) for i in range(2, 2 * N, 2)]

# Solve and print the result
print(max_happiness(N, W, items))