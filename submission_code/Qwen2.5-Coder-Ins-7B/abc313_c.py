# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Sort the array to easily find the minimum and maximum
A.sort()

# The goal is to make the difference between the minimum and maximum at most 1
# We can achieve this by moving elements to the median of the array
# The median minimizes the sum of absolute deviations

# Find the median
median = A[N // 2]

# Calculate the number of operations needed
operations = 0
for num in A:
    if num < median:
        operations += median - num
    elif num > median + 1:
        operations += num - (median + 1)

print(operations)