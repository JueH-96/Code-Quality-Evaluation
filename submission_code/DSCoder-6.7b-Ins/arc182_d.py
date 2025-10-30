import sys
from collections import deque

def min_operations(n, m, a, b):
    if a == b:
        return 0

    # Create a deque to store the differences between A and B
    diff = deque()
    for i in range(n):
        diff.append((b[i] - a[i]) % m)

    # Calculate the minimum number of operations
    operations = 0
    while diff:
        # Pop the first element from the deque
        d = diff.popleft()

        # If the difference is greater than M/2, we need to subtract from the first element
        if d > m // 2:
            operations += d
            if diff:
                diff[0] = (diff[0] + m - d) % m
        # Otherwise, we need to add to the first element
        else:
            operations += d
            if diff:
                diff[0] = (diff[0] + d) % m

    return operations

# Read the inputs
n, m = map(int, sys.stdin.readline().split())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))

# Print the minimum number of operations
print(min_operations(n, m, a, b))