def count_operations(N, A):
    operations = 0
    while True:
        # Filter out zeros
        A = [a for a in A if a > 0]
        if len(A) <= 1:
            break
        # Sort in descending order
        A.sort(reverse=True)
        # Decrease the top two elements by 1
        A[0] -= 1
        A[1] -= 1
        operations += 1
    return operations

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute and print the result
print(count_operations(N, A))