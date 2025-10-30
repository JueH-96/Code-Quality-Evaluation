n = int(input())
steps = [tuple(input().split()) for _ in range(n)]
steps = [(int(a), s) for a, s in steps]

if not steps:
    print(0)
    exit()

current_a, current_s = steps[0]
prev_dp = {}
if current_s == 'L':
    # Left is at current_a, right can be any position with cost 0
    prev_dp = {r: 0 for r in range(1, 101)}
else:
    # Right is at current_a, left can be any position with cost 0
    prev_dp = {l: 0 for l in range(1, 101)}
prev_type = current_s

for i in range(1, n):
    current_a, current_s = steps[i]
    new_dp = {}
    prev_a, prev_s = steps[i-1]
    for prev_pos in prev_dp:
        cost_prev = prev_dp[prev_pos]
        if current_s == 'L':
            if prev_s == 'L':
                # Previous step was L, prev_pos is right hand's position
                movement = abs(current_a - prev_a)
                new_pos = prev_pos
                new_cost = cost_prev + movement
                if new_pos in new_dp:
                    if new_cost < new_dp[new_pos]:
                        new_dp[new_pos] = new_cost
                else:
                    new_dp[new_pos] = new_cost
            else:
                # Previous step was R, prev_pos is left hand's position
                movement = abs(current_a - prev_pos)
                new_pos = prev_a
                new_cost = cost_prev + movement
                if new_pos in new_dp:
                    if new_cost < new_dp[new_pos]:
                        new_dp[new_pos] = new_cost
                else:
                    new_dp[new_pos] = new_cost
        else:  # current_s == 'R'
            if prev_s == 'L':
                # Previous step was L, prev_pos is right hand's position
                movement = abs(current_a - prev_pos)
                new_pos = prev_a
                new_cost = cost_prev + movement
                if new_pos in new_dp:
                    if new_cost < new_dp[new_pos]:
                        new_dp[new_pos] = new_cost
                else:
                    new_dp[new_pos] = new_cost
            else:
                # Previous step was R, prev_pos is left hand's position
                movement = abs(current_a - prev_a)
                new_pos = prev_pos
                new_cost = cost_prev + movement
                if new_pos in new_dp:
                    if new_cost < new_dp[new_pos]:
                        new_dp[new_pos] = new_cost
                else:
                    new_dp[new_pos] = new_cost
    prev_dp = new_dp
    prev_type = current_s

print(min(prev_dp.values()))