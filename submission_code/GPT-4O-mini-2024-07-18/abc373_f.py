def max_happiness(N, W, items):
    dp = [0] * (W + 1)

    for w, v in items:
        for k in range(1, W // w + 1):
            happiness = k * v - k * k
            weight = k * w
            for j in range(W, weight - 1, -1):
                dp[j] = max(dp[j], dp[j - weight] + happiness)

    return max(dp)

import sys
input = sys.stdin.read
data = input().splitlines()

N, W = map(int, data[0].split())
items = [tuple(map(int, line.split())) for line in data[1:N + 1]]

result = max_happiness(N, W, items)
print(result)