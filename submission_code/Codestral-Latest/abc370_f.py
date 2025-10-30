import sys
from itertools import accumulate

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Calculate the prefix sums of the masses
prefix_sums = list(accumulate(A))

# Function to find the minimum sum of a subarray of length at least N/K
def min_subarray_sum(prefix_sums, N, K):
    min_sum = float('inf')
    for i in range(N - K + 1):
        if i == 0:
            subarray_sum = prefix_sums[i + K - 1]
        else:
            subarray_sum = prefix_sums[i + K - 1] - prefix_sums[i - 1]
        min_sum = min(min_sum, subarray_sum)
    return min_sum

# Function to find the number of cut lines that are never cut
def count_uncut_lines(A, N, K, min_sum):
    total_sum = sum(A)
    target_sum = total_sum // K
    if total_sum % K != 0:
        target_sum += 1

    current_sum = 0
    uncut_lines = 0
    for i in range(N):
        current_sum += A[i]
        if current_sum >= target_sum:
            current_sum = 0
            uncut_lines += 1

    return uncut_lines

# Find the minimum sum of a subarray of length at least N/K
min_sum = min_subarray_sum(prefix_sums, N, N // K)

# Find the number of cut lines that are never cut
uncut_lines = count_uncut_lines(A, N, K, min_sum)

print(min_sum, uncut_lines)