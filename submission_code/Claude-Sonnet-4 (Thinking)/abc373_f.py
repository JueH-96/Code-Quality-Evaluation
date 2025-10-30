N, W = map(int, input().split())
items = []
for _ in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

# dp[w] = maximum happiness with total weight at most w
dp = [0] * (W + 1)

for w_i, v_i in items:
    new_dp = dp[:]
    max_k = W // w_i
    for k in range(1, max_k + 1):
        weight = k * w_i
        happiness = k * v_i - k * k
        if happiness <= 0:
            break
        for w in range(weight, W + 1):
            new_dp[w] = max(new_dp[w], dp[w - weight] + happiness)
    dp = new_dp

print(dp[W])