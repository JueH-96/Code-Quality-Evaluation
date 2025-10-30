import sys
import bisect

# Read all input and split into a list of strings
data = sys.stdin.read().split()
index = 0

# Read N, M, P
N, M, P = int(data[index]), int(data[index + 1]), int(data[index + 2])
index += 3

# Read A list
A = [int(data[i]) for i in range(index, index + N)]
index += N

# Read B list
B = [int(data[i]) for i in range(index, index + M)]

# Sort B in ascending order
B_sorted = sorted(B)

# Compute prefix sum of B_sorted
prefix_sum_B = [0]
cum_sum = 0
for b in B_sorted:
    cum_sum += b
    prefix_sum_B.append(cum_sum)

# Initialize total sum
total_sum = 0

# For each A_i, compute the sum over all B_j
for a in A:
    T = P - a
    count_leq = bisect.bisect_right(B_sorted, T)
    sum_B_leq = prefix_sum_B[count_leq]
    sum_j = sum_B_leq + a * count_leq + P * (M - count_leq)
    total_sum += sum_j

# Output the total sum
print(total_sum)