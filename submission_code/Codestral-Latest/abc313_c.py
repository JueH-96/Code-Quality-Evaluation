import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Sort the array to easily find the min and max values
A.sort()

# Use a deque to efficiently pop elements from both ends
dq = deque(A)

# Initialize the number of operations
operations = 0

# While the difference between the max and min is more than 1
while dq[-1] - dq[0] > 1:
    # Decrease the max value and increase the min value
    dq[-1] -= 1
    dq[0] += 1

    # If the max value becomes less than the second max, pop it
    if dq[-1] < dq[-2]:
        dq.pop()

    # If the min value becomes more than the second min, pop it
    if dq[0] > dq[1]:
        dq.popleft()

    # Increment the number of operations
    operations += 1

# Print the number of operations
print(operations)