import sys
import bisect

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read N and Q
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Read R values
R = [int(data[index + i]) for i in range(N)]
index += N

# Sort R in ascending order
sorted_R = sorted(R)

# Compute prefix sum array, including sum for 0 sleighs
prefix_sum = [0]
cum_sum = 0
for val in sorted_R:
    cum_sum += val
    prefix_sum.append(cum_sum)

# Read all X queries
X_list = [int(data[index + i]) for i in range(Q)]

# Compute answers for each query
answers = []
for X in X_list:
    # Binary search to find the smallest index i where prefix_sum[i] > X
    i = bisect.bisect_right(prefix_sum, X)
    # The largest M where prefix_sum[M] <= X is i - 1
    M = i - 1
    answers.append(str(M))

# Output all answers, one per line
sys.stdout.write('
'.join(answers) + '
')