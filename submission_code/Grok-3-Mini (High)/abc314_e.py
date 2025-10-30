import sys
import math

# Read all input data
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read the wheels
wheels = []
for _ in range(N):
    C = int(data[index])
    index += 1
    P = int(data[index])
    index += 1
    S_list = [int(data[index + k]) for k in range(P)]
    index += P
    wheels.append((C, S_list))

# Policy iteration
# Initialize policy: all states choose wheel 0
policy = [0] * M

while True:
    # Policy evaluation: compute dp for current policy
    dp = [0.0] * (M + 1)  # dp[x] for x from 0 to M, dp[M] = 0
    for x in range(M - 1, -1, -1):  # From M-1 down to 0
        i = policy[x]
        C_i, S_list = wheels[i]
        P_i = len(S_list)
        num_zero = 0
        sum_K = 0.0  # Sum of dp[y] for s > 0
        for s in S_list:
            if s == 0:
                num_zero += 1
            else:
                y = min(x + s, M)
                sum_K += dp[y]
        p_stay = num_zero / float(P_i)
        coef = 1.0 - p_stay
        rhs = C_i + (sum_K / float(P_i))
        dp[x] = rhs / coef
    
    # Policy improvement: find best action for each state using current dp
    new_policy = [0] * M
    for x in range(M):
        min_Q = float('inf')
        best_i = -1
        for i_wheel in range(N):
            C, S_list_w = wheels[i_wheel]
            P_w = len(S_list_w)
            sum_Q_sum = 0.0
            for s in S_list_w:
                y = min(x + s, M)
                sum_Q_sum += dp[y]
            expected_dp = sum_Q_sum / float(P_w)
            Q_value = C + expected_dp
            if Q_value < min_Q:
                min_Q = Q_value
                best_i = i_wheel
        new_policy[x] = best_i
    
    # Check if policy changed
    if new_policy == policy:
        break  # Policy stable, optimal
    else:
        policy = new_policy  # Update policy

# Output the expected cost starting from 0 points
print(f"{dp[0]:.10f}")