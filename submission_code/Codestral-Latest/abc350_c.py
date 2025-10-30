# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Create a dictionary to store the positions of each element
pos = {v: i for i, v in enumerate(A)}

operations = []

# Sort the permutation by performing the required swaps
for i in range(N):
    if A[i] != i + 1:
        # Find the position of the correct element
        correct_pos = pos[i + 1]
        # Swap the elements
        A[i], A[correct_pos] = A[correct_pos], A[i]
        # Update the positions
        pos[A[correct_pos]] = correct_pos
        pos[A[i]] = i
        # Record the operation
        operations.append((i + 1, correct_pos + 1))

# Print the number of operations
print(len(operations))
# Print the operations
for op in operations:
    print(*op)