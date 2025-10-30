# YOUR CODE HERE
import sys
from collections import deque

# Read all input data
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
K = data[index]
index += 1
P = data[index:index + N]

# Create position array (1-based index)
pos = [0] * (N + 1)
for i in range(N):
    val = P[i]
    pos[val] = i + 1  # Store 1-based position

# Create array of positions in order of values
arr = [pos[val] for val in range(1, N + 1)]

# Sliding window minimum
deq_min = deque()
min_win = [0] * (N - K + 1)
for i in range(N):
    # Remove indices out of the current window
    if deq_min and deq_min[0] <= i - K:
        deq_min.popleft()
    # Remove indices with larger values from the back
    while deq_min and arr[deq_min[-1]] > arr[i]:
        deq_min.pop()
    # Add current index
    deq_min.append(i)
    # If window is full, record the minimum
    if i >= K - 1:
        start_idx = i - K + 1
        min_win[start_idx] = arr[deq_min[0]]

# Sliding window maximum
deq_max = deque()
max_win = [0] * (N - K + 1)
for i in range(N):
    # Remove indices out of the current window
    if deq_max and deq_max[0] <= i - K:
        deq_max.popleft()
    # Remove indices with smaller values from the back
    while deq_max and arr[deq_max[-1]] < arr[i]:
        deq_max.pop()
    # Add current index
    deq_max.append(i)
    # If window is full, record the maximum
    if i >= K - 1:
        start_idx = i - K + 1
        max_win[start_idx] = arr[deq_max[0]]

# Compute the minimum difference of max and min in each window
ans = min(max_win[i] - min_win[i] for i in range(N - K + 1))

# Output the answer
print(ans)