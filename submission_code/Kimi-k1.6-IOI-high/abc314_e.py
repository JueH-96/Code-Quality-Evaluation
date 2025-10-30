n, m = map(int, input().split())
roulettes = []
for _ in range(n):
    parts = list(map(int, input().split()))
    c = parts[0]
    p = parts[1]
    s_list = parts[2:]
    roulettes.append((c, p, s_list))

dp = [0.0] * m

for k in range(m-1, -1, -1):
    min_cost = float('inf')
    for (c, p, s_list) in roulettes:
        zero_count = sum(1 for s in s_list if s == 0)
        q_i = (p - zero_count) / p
        sum_pos = 0.0
        for s in s_list:
            if s <= 0:
                continue
            next_k = k + s
            if next_k >= m:
                contrib = 0.0
            else:
                contrib = dp[next_k]
            sum_pos += (1.0 / p) * contrib
        expected_cost = (c + sum_pos) / q_i
        if expected_cost < min_cost:
            min_cost = expected_cost
    dp[k] = min_cost

# Print with sufficient precision
print("{0:.10f}".format(dp[0]))