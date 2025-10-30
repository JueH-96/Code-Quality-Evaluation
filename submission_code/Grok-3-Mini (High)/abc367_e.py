import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1

# Read X values
X_val = [0] * (N + 1)
for i in range(1, N + 1):
    X_val[i] = int(data[index])
    index += 1

# Read A values
A_val = [0] * (N + 1)
for i in range(1, N + 1):
    A_val[i] = int(data[index])
    index += 1

# Precompute jump table for binary lifting
MAX_LOG = 60  # Since 2^60 > 10^18
jump = [[0] * MAX_LOG for _ in range(N + 1)]

# Set the first level of jump table: jump[node][0] = X_val[node]
for node in range(1, N + 1):
    jump[node][0] = X_val[node]

# Compute higher levels of jump table
for b in range(1, MAX_LOG):
    for node in range(1, N + 1):
        prev = jump[node][b - 1]
        jump[node][b] = jump[prev][b - 1]

# Compute sigma^K(i) for each i and get the corresponding A value
ans = []
for i in range(1, N + 1):
    current = i
    for b in range(MAX_LOG):
        if (K >> b) & 1:  # If the b-th bit of K is set
            current = jump[current][b]
    # current is now sigma^K(i)
    val = A_val[current]
    ans.append(val)

# Output the result
print(' '.join(map(str, ans)))