import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1
Y = int(data[index])
index += 1
A = [0] * N
B = [0] * N
for i in range(N):
    A[i] = int(data[index])
    index += 1
    B[i] = int(data[index])
    index += 1

INF = 1000000005

# Initialize prev_dp
prev_dp = [[INF for _ in range(Y + 1)] for _ in range(N + 1)]
prev_dp[0][0] = 0

# Add each dish one by one
for dish_idx in range(N):
    A_i = A[dish_idx]
    B_i = B[dish_idx]
    curr_dp = [[INF for _ in range(Y + 1)] for _ in range(N + 1)]
    for k in range(0, N + 1):
        for sB in range(0, Y + 1):
            # Not taking the dish
            curr_dp[k][sB] = prev_dp[k][sB]
            # Taking the dish, if possible
            if k >= 1 and sB >= B_i:
                prev_sum_A = prev_dp[k - 1][sB - B_i]
                if prev_sum_A < INF:
                    curr_dp[k][sB] = min(curr_dp[k][sB], prev_sum_A + A_i)
    # Update prev_dp to curr_dp
    prev_dp = curr_dp

# Now prev_dp is the dp after considering all dishes
# Find the largest k such that there exists sB <= Y with prev_dp[k][sB] <= X
m = -1
for k in range(N, -1, -1):
    possible = False
    for sB in range(Y + 1):
        if prev_dp[k][sB] <= X:
            possible = True
            break
    if possible:
        m = k
        break

# The maximum number of dishes Snuke eats is min(N, m + 1)
answer = min(N, m + 1)
print(answer)