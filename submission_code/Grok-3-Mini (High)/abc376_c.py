import bisect
import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N - 1]))

# Sort the lists
A_sorted = sorted(A)
B_sorted = sorted(B)

# Initialize variables
impossible_flag = False
max_req = 0

# Check each k from 1 to N
for k in range(1, N + 1):
    val = A_sorted[k - 1]
    num_small = bisect.bisect_left(B_sorted, val)
    if num_small >= k:
        impossible_flag = True
    elif num_small == k - 1:
        if val > max_req:
            max_req = val

# Output the result
if impossible_flag:
    print(-1)
else:
    print(max_req)