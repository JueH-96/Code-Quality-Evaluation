import sys

# Read input
# Read N and M
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])

# Read black ball values
B = list(map(int, sys.stdin.readline().split()))

# Read white ball values
W = list(map(int, sys.stdin.readline().split()))

# Sort B and W in descending order
# We want to pick the balls with the largest values first to maximize the sum.
B.sort(reverse=True)
W.sort(reverse=True)

# Compute prefix sums for B
# P_B[k] stores the sum of the top k largest values in B (0-indexed list B)
# P_B[0] = 0 (sum of 0 elements)
# P_B[k] = B[0] + ... + B[k-1] for k >= 1
P_B = [0] * (N + 1)
for i in range(N):
    P_B[i+1] = P_B[i] + B[i]

# Compute prefix sums for W
# P_W[k] stores the sum of the top k largest values in W (0-indexed list W)
# P_W[0] = 0 (sum of 0 elements)
# P_W[k] = W[0] + ... + W[k-1] for k >= 1
P_W = [0] * (M + 1)
for i in range(M):
    P_W[i+1] = P_W[i] + W[i]

# Compute prefix maximums for P_W
# M_W[k] stores the maximum sum achievable by choosing j white balls,
# where 0 <= j <= k. This is max(P_W[0], P_W[1], ..., P_W[k]).
# Since values can be negative, taking fewer than the maximum allowed balls
# might yield a higher sum. M_W helps find the maximum sum for choosing *at most* k white balls.
M_W = [0] * (M + 1)
M_W[0] = P_W[0] # Which is 0, the sum of choosing 0 white balls
for i in range(1, M + 1):
    M_W[i] = max(M_W[i-1], P_W[i])

# Initialize maximum sum
# Choosing 0 black balls and 0 white balls is always allowed, sum is 0.
# This case corresponds to k_b=0 and k_w=0. The constraint k_b >= k_w is 0 >= 0, satisfied.
# The sum is P_B[0] + P_W[0] = 0 + 0 = 0.
# Using the logic from the loop for k_b=0: k_w_limit = min(0, M) = 0.
# The maximum white ball sum for k_w <= 0 is M_W[0] = P_W[0] = 0.
# The total sum for k_b=0 is P_B[0] + M_W[0] = 0 + 0 = 0.
# So, initializing max_sum = 0 correctly captures the k_b = 0 case.
max_sum = 0

# Iterate through possible numbers of black balls chosen (k_b)
# k_b can range from 0 to N. We iterate from k_b = 1 to N, and the k_b = 0 case is handled by initialization.
for k_b in range(1, N + 1):
    # For a fixed number of black balls k_b (we assume the top k_b are chosen, sum P_B[k_b]),
    # we can choose any number of white balls k_w such that:
    # 1. 0 <= k_w <= M (cannot choose more white balls than available)
    # 2. k_b >= k_w (the problem constraint)
    # Combining these, the valid range for k_w is 0 <= k_w <= min(k_b, M).
    
    # To maximize the total sum = P_B[k_b] + (sum of k_w white balls) for this fixed k_b,
    # we need to find the number of white balls k_w in [0, min(k_b, M)] that maximizes their sum.
    # Since we must choose the top k_w balls to maximize sum for a fixed k_w,
    # the sum is P_W[k_w]. The maximum of P_W[k_w] for k_w in [0, k_w_limit]
    # is given by M_W[k_w_limit], where k_w_limit = min(k_b, M).
    
    k_w_limit = min(k_b, M)
    
    # Calculate the potential maximum sum for choosing exactly k_b black balls
    # (the top k_b) and the optimal subset of white balls (with count <= k_w_limit).
    current_sum = P_B[k_b] + M_W[k_w_limit]
    
    # Update the overall maximum sum found so far.
    max_sum = max(max_sum, current_sum)

# Print the result
print(max_sum)