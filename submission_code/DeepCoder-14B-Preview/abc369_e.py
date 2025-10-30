n = int(input())
actions = []
for _ in range(n):
    a, s = input().split()
    a = int(a)
    actions.append((a, s))

# Initialize the DP table for step 0 (all possible initial positions with 0 fatigue)
prev_dp = [[0 for _ in range(101)] for __ in range(101)]

for step in range(n):
    current_dp = [[float('inf') for _ in range(101)] for __ in range(101)]
    a, s = actions[step]
    for l_prev in range(1, 101):
        for r_prev in range(1, 101):
            current_cost = prev_dp[l_prev][r_prev]
            if current_cost == float('inf'):
                continue
            if s == 'L':
                new_l = a
                new_r = r_prev
                cost = abs(new_l - l_prev)
                total_cost = current_cost + cost
                if total_cost < current_dp[new_l][new_r]:
                    current_dp[new_l][new_r] = total_cost
            else:
                new_r = a
                new_l = l_prev
                cost = abs(new_r - r_prev)
                total_cost = current_cost + cost
                if total_cost < current_dp[new_l][new_r]:
                    current_dp[new_l][new_r] = total_cost
    prev_dp = current_dp

# Find the minimal fatigue in the final DP state
min_fatigue = float('inf')
for l in range(1, 101):
    for r in range(1, 101):
        if prev_dp[l][r] < min_fatigue:
            min_fatigue = prev_dp[l][r]

print(min_fatigue)