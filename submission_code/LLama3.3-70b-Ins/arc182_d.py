import sys

def min_operations(N, M, A, B):
    """
    Calculate the minimum number of operations required to make A equal to B.

    Args:
    N (int): The length of the sequences A and B.
    M (int): The maximum value of the elements in A and B.
    A (list): The initial sequence A.
    B (list): The target sequence B.

    Returns:
    int: The minimum number of operations required to make A equal to B. Returns -1 if it's impossible.
    """
    operations = 0
    for i in range(N):
        # Calculate the difference between A[i] and B[i]
        diff = (B[i] - A[i]) % M
        # If the difference is greater than M/2, it's shorter to go the other way around
        if diff > M // 2:
            diff = M - diff
        # If the difference is not 0, we need to perform operations
        if diff != 0:
            # If the previous element is the same as the target element, we can't perform the operation
            if i > 0 and (A[i-1] + diff) % M == B[i-1]:
                return -1
            # If the next element is the same as the target element, we can't perform the operation
            if i < N-1 and (A[i+1] + diff) % M == B[i+1]:
                return -1
            # Perform the operations
            operations += diff
            # Update A[i]
            A[i] = (A[i] + diff) % M
    return operations

# Read the input
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Calculate the minimum number of operations
min_ops = min_operations(N, M, A, B)

# Print the result
print(min_ops)