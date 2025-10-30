import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S_str = data[index]
index += 1
C_str_list = data[index:index + N]
S_bits = [int(c) for c in S_str]
C_cost = [int(c) for c in C_str_list]

# Compute cumulative sums
cum = [[[0 for _ in range(N + 1)] for _ in range(2)] for _ in range(2)]  # cum[p][s][k]
for k in range(1, N + 1):  # k is 1-based position
    parity_k = k % 2
    s_val_k = S_bits[k - 1]
    for p in range(2):
        for s in range(2):
            cum[p][s][k] = cum[p][s][k - 1]
            if (k % 2 == p) and (s_val_k == s):
                cum[p][s][k] += C_cost[k - 1]

# Initialize answer
ans = float('inf')

# Iterate over each possible i
for i in range(1, N):  # i from 1 to N-1
    P = i % 2
    # Compute cost for V=0
    cost_V0 = (cum[P][1][i] + (cum[1 - P][1][N] - cum[1 - P][1][i]) + 
               cum[1 - P][0][i] + (cum[P][0][N] - cum[P][0][i]))
    # Compute cost for V=1
    cost_V1 = (cum[P][0][i] + (cum[1 - P][0][N] - cum[1 - P][0][i]) + 
               cum[1 - P][1][i] + (cum[P][1][N] - cum[P][1][i]))
    # Minimum cost for this i
    min_cost_i = min(cost_V0, cost_V1)
    # Update answer
    if min_cost_i < ans:
        ans = min_cost_i

# Output the answer
print(int(ans))  # Cast to int to ensure integer output