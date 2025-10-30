def read_ints():
    return list(map(int, input().split()))

def max_happiness(N, W, items):
    # Initialize DP table with -infinity
    dp = [[float('-inf')] * (W + 1) for _ in range(N + 1)]
    dp[0][0] = 0  # Base case: no items, no weight, happiness is 0

    for i in range(1, N + 1):
        w, v = items[i - 1]
        for j in range(W + 1):
            # Case 1: Do not take any item of this type
            dp[i][j] = max(dp[i][j], dp[i - 1][j])
            # Case 2: Take k items of this type (k from 1 to 10^10 theoretically)
            # But we only need to consider up to min(j // w, sqrt(2*v)) due to the happiness function
            for k in range(1, min(j // w, int((2 * v) ** 0.5)) + 1):
                dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w] + k * v - k * k)

    return max(dp[N])

# Read input
N, W = read_ints()
items = [read_ints() for _ in range(N)]

# Calculate and print the answer
print(max_happiness(N, W, items))