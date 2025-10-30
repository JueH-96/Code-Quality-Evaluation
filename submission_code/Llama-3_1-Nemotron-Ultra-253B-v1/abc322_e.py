n, k, p = map(int, input().split())
plans = []
for _ in range(n):
    parts = list(map(int, input().split()))
    c = parts[0]
    a = parts[1:k+1]
    plans.append((c, a))

current_dp = {}
initial_state = tuple([0] * k)
current_dp[initial_state] = 0

for (cost, a) in plans:
    temp_dp = {}
    for state in current_dp:
        new_state = list(state)
        for j in range(k):
            new_state[j] = min(new_state[j] + a[j], p)
        new_state = tuple(new_state)
        new_cost = current_dp[state] + cost
        if new_state in temp_dp:
            if new_cost < temp_dp[new_state]:
                temp_dp[new_state] = new_cost
        else:
            temp_dp[new_state] = new_cost
    for state in temp_dp:
        if state in current_dp:
            if temp_dp[state] < current_dp[state]:
                current_dp[state] = temp_dp[state]
        else:
            current_dp[state] = temp_dp[state]

min_total = float('inf')
for state in current_dp:
    if all(x >= p for x in state):
        if current_dp[state] < min_total:
            min_total = current_dp[state]

print(min_total if min_total != float('inf') else -1)