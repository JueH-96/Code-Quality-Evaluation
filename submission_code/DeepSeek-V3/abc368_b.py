# YOUR CODE HERE
def count_operations(N, A):
    operations = 0
    while True:
        # Sort in descending order
        A_sorted = sorted(A, reverse=True)
        # Check if there are at least two positive elements
        if A_sorted[0] <= 0 or A_sorted[1] <= 0:
            break
        # Decrease the top two elements by 1
        A_sorted[0] -= 1
        A_sorted[1] -= 1
        operations += 1
        # Update the original list
        A = A_sorted
    return operations

# Read input
N = int(input())
A = list(map(int, input().split()))
# Compute the number of operations
result = count_operations(N, A)
# Print the result
print(result)