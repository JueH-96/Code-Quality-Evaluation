# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the position of each element in A
pos = {A[i]: i for i in range(N)}

# Initialize the list of operations
operations = []

# Iterate over the elements in A
for i in range(N):
    # If the current element is not in the correct position
    if A[i] != i + 1:
        # Find the position of the correct element
        j = pos[i + 1]

        # Swap the current element with the correct element
        A[i], A[j] = A[j], A[i]

        # Update the positions of the swapped elements
        pos[A[i]], pos[A[j]] = pos[A[j]], pos[A[i]]

        # Add the operation to the list of operations
        operations.append((i + 1, j + 1))

# Print the number of operations
print(len(operations))

# Print the operations
for operation in operations:
    print(operation[0], operation[1])