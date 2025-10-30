import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
H = [0] * (N + 1)
for i in range(1, N + 1):
    H[i] = data[index]
    index += 1

# Compute R_k: nearest greater to the left index
R = [0] * (N + 1)
stack = []
for k in range(1, N + 1):
    while stack and H[stack[-1]] <= H[k]:
        stack.pop()
    if not stack:
        R[k] = 0
    else:
        R[k] = stack[-1]
    stack.append(k)

# Now compute c_i using difference array
delta = [0] * (N + 1)  # delta[0] unused
for k in range(2, N + 1):
    L = max(1, R[k])
    R_right = k - 1
    delta[L] += 1
    delta[R_right + 1] -= 1

# Compute prefix sum to get ans
ans = [0] * (N + 1)
cum = 0
for i in range(1, N + 1):
    cum += delta[i]
    ans[i] = cum

# Output the result
print(' '.join(map(str, [ans[i] for i in range(1, N + 1)])))