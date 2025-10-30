import math

def is_square(n):
    """
    Checks if a non-negative integer n is a square number.
    """
    return int(math.sqrt(n)) ** 2 == n

def count_square_pairs(A):
    """
    Counts the number of pairs (i, j) such that 1 <= i < j <= N and A[i] * A[j] is a square number.
    """
    N = len(A)
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if is_square(A[i] * A[j]):
                count += 1
    return count

# Read input from stdin
N = int(input())
A = [int(x) for x in input().split()]

# Solve the problem
result = count_square_pairs(A)

# Print the answer to stdout
print(result)