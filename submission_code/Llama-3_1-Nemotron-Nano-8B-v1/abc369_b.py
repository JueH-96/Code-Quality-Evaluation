n = int(input())
steps = []
for _ in range(n):
    a, s = input().split()
    steps.append((int(a), s))

INF = float('inf')
# Initialize DP for step 0 (initial positions)
prev_dp = [[0] * 101 for _ in range(101)]

for i in range(n):
    a, s = steps[i]
    current_dp = [[INF] * 101 for _ in range(101)]
    for l_prev in range(1, 101):
        for r_prev in range(1, 101):
            if prev_dp[l_prev][r_prev] == INF:
                continue
            if s == 'L':
                new_l = a
                new_r = r_prev
                cost = abs(a - l_prev)
            else:
                new_l = l_prev
                new_r = a
                cost = abs(a - r_prev)
            if current_dp[new_l][new_r] > prev_dp[l_prev][r_prev] + cost:
                current_dp[new_l][new_r] = prev_dp[l_prev][r_prev] + cost
    prev_dp = current_dp

min_fatigue = min(prev_dp[l][r] for l in range(1, 101) for r in range(1, 101))
print(min_fatigue)