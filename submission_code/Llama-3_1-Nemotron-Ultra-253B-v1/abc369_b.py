n = int(input())
presses = []
for _ in range(n):
    a, s = input().split()
    presses.append((int(a), s))

INF = float('inf')
current_dp = [[INF] * 101 for _ in range(101)]

first_a, first_s = presses[0]
if first_s == 'L':
    for r in range(1, 101):
        current_dp[first_a][r] = 0
else:
    for l in range(1, 101):
        current_dp[l][first_a] = 0

for i in range(1, n):
    a, s = presses[i]
    next_dp = [[INF] * 101 for _ in range(101)]
    for l_prev in range(1, 101):
        for r_prev in range(1, 101):
            if current_dp[l_prev][r_prev] == INF:
                continue
            cost = current_dp[l_prev][r_prev]
            if s == 'L':
                new_l = a
                new_r = r_prev
                movement = abs(new_l - l_prev)
                new_cost = cost + movement
                if new_cost < next_dp[new_l][new_r]:
                    next_dp[new_l][new_r] = new_cost
            else:
                new_r = a
                new_l = l_prev
                movement = abs(new_r - r_prev)
                new_cost = cost + movement
                if new_cost < next_dp[new_l][new_r]:
                    next_dp[new_l][new_r] = new_cost
    current_dp = next_dp

min_fatigue = INF
for l in range(1, 101):
    for r in range(1, 101):
        if current_dp[l][r] < min_fatigue:
            min_fatigue = current_dp[l][r]

print(min_fatigue)