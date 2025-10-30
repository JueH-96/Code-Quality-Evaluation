import sys
data = sys.stdin.read().split()
N = int(data[0])

# Collect keys for left and right hands
left_keys = []
right_keys = []
for i in range(N):
    A = int(data[2 * i + 1])
    S = data[2 * i + 2]
    if S == 'L':
        left_keys.append(A)
    elif S == 'R':
        right_keys.append(A)

# Compute cost for left hand
if len(left_keys) > 1:
    cost_left = sum(abs(left_keys[j] - left_keys[j - 1]) for j in range(1, len(left_keys)))
else:
    cost_left = 0

# Compute cost for right hand
if len(right_keys) > 1:
    cost_right = sum(abs(right_keys[j] - right_keys[j - 1]) for j in range(1, len(right_keys)))
else:
    cost_right = 0

# Total cost
total_cost = cost_left + cost_right

# Output the result
print(total_cost)