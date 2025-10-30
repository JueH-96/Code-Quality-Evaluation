import sys
import bisect

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Initialize answer
ans = 0

# Iterate through each mochi as the top
for size in A:
    # Calculate the minimum size required for the bottom mochi
    val = 2 * size
    # Find the leftmost index where A[j] >= val using binary search
    idx = bisect.bisect_left(A, val)
    # Number of bottom mochi that satisfy the condition
    num_j = N - idx
    # Add to the answer
    ans += num_j

# Output the result
print(ans)