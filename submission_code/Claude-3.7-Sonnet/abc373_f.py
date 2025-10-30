def max_happiness(N, W, weights, values):
    # DP[w] represents the maximum happiness possible by considering 
    # the first i-1 types of items and a weight of at most w.
    DP = [0] * (W + 1)
    
    for i in range(1, N + 1):
        next_DP = DP.copy()
        for w in range(W + 1):
            # Consider taking 1, 2, ... items of the current type
            for k in range(1, min((values[i-1] + 1) // 2 + 1, w // weights[i-1] + 1)):
                happiness = k * values[i-1] - k * k
                next_DP[w] = max(next_DP[w], DP[w - k * weights[i-1]] + happiness)
        
        DP = next_DP
    
    return DP[W]

# Read the input
N, W = map(int, input().split())
weights = []
values = []
for _ in range(N):
    w, v = map(int, input().split())
    weights.append(w)
    values.append(v)

# Calculate and print the answer
result = max_happiness(N, W, weights, values)
print(result)