import sys
from collections import deque

input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1

results = []

for _ in range(T):
    N = int(data[index])
    index += 1
    P = list(map(int, data[index:index + N]))
    index += N

    # Create a deque to store the positions of elements
    positions = deque([0] * (N + 1))
    for i in range(N):
        positions[P[i]] = i

    # Initialize the number of operations
    operations = 0

    # Process the permutation
    for i in range(1, N + 1):
        if positions[i] != i - 1:
            # Find the position of i
            pos = positions[i]

            # Perform the operation
            if pos > 0:
                operations += 1
                # Update the positions
                for j in range(pos + 1):
                    positions[P[j]] = j
                P[:pos + 1] = sorted(P[:pos + 1])

            if pos < N - 1:
                operations += 1
                # Update the positions
                for j in range(pos + 1, N):
                    positions[P[j]] = j
                P[pos + 1:] = sorted(P[pos + 1:])

    results.append(operations)

# Print the results
for result in results:
    print(result)