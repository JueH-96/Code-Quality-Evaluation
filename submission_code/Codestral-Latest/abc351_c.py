import sys
from collections import deque

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Initialize the sequence
sequence = deque()

# Process each ball
for i in range(N):
    # Add the i-th ball to the sequence
    sequence.append(2 ** A[i])

    # Process the sequence until it stabilizes
    while len(sequence) > 1:
        if sequence[-1] != sequence[-2]:
            break
        else:
            # Remove the last two balls and add their sum
            size_sum = sequence.pop() + sequence.pop()
            sequence.append(size_sum)

# Output the number of balls remaining in the sequence
print(len(sequence))