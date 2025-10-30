import sys
import bisect

# Read input
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

# Sort the array
sorted_A = sorted(A)

# Compute prefix sum of sorted array
prefix_sum = [0]
cum_sum = 0
for num in sorted_A:
    cum_sum += num
    prefix_sum.append(cum_sum)
total_sum = prefix_sum[-1]

# Compute the sum of elements greater than each A_i
results = []
for val in A:
    # Find the insertion point for val to get the index where values > val start
    idx = bisect.bisect_right(sorted_A, val)
    # Sum of elements from idx to end is total_sum - prefix_sum[idx]
    sum_greater = total_sum - prefix_sum[idx]
    results.append(sum_greater)

# Output the results separated by spaces
print(*results)