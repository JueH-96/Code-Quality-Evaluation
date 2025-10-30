import sys
import math

# Read all input
data = sys.stdin.read().split()
index = 0

# Read N, K, P
N = int(data[index])
index += 1
K = int(data[index])
index += 1
P = int(data[index])
index += 1

# Read the N plans: cost and A values
cost = []
A_list = []
for i in range(N):
    C_i = int(data[index])
    index += 1
    A_i = []
    for j in range(K):
        A_ij = int(data[index])
        A_i.append(A_ij)
        index += 1
    A_list.append(tuple(A_i))
    cost.append(C_i)

# Define helper functions
def get_conf(idx, base, K):
    return tuple((idx // (base ** j)) % base for j in range(K))

def get_index(conf, base):
    index_val = 0
    for j in range(len(conf)):
        index_val += conf[j] * (base ** j)
    return index_val

# Compute base and number of configurations
base = P + 1
num_conf = base ** K

# Initialize DP table
dp = [[float('inf') for _ in range(num_conf)] for _ in range(N + 1)]

# Initial state: no plans used, all sums are 0
conf_init = tuple([0] * K)
idx_init = get_index(conf_init, base)
dp[0][idx_init] = 0

# DP transitions
for i in range(N):  # for each plan
    for current_idx in range(num_conf):
        if dp[i][current_idx] < float('inf'):
            # Option 1: do not use plan i
            dp[i + 1][current_idx] = min(dp[i + 1][current_idx], dp[i][current_idx])
            
            # Option 2: use plan i
            current_conf = get_conf(current_idx, base, K)
            new_conf = tuple(min(current_conf[j] + A_list[i][j], P) for j in range(K))
            new_idx = get_index(new_conf, base)
            dp[i + 1][new_idx] = min(dp[i + 1][new_idx], dp[i][current_idx] + cost[i])

# Target configuration: all parameters at least P, so capped sum is P for all
target_conf = tuple([P] * K)
target_idx = get_index(target_conf, base)
cost_min = dp[N][target_idx]

# Output the result
if cost_min == float('inf'):
    print(-1)
else:
    print(int(cost_min))