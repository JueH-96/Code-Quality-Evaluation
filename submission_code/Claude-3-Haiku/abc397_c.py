from collections import defaultdict

# Read input
N = int(input())
A = list(map(int, input().split()))

# Function to count distinct elements in a subarray
def count_distinct(arr):
    count = defaultdict(int)
    for x in arr:
        count[x] += 1
    return len(count)

# Find the maximum sum of distinct elements in the two subarrays
max_sum = 0
for i in range(1, N):
    left_count = count_distinct(A[:i])
    right_count = count_distinct(A[i:])
    max_sum = max(max_sum, left_count + right_count)

# Print the answer
print(max_sum)