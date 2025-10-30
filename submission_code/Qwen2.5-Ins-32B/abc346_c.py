import sys

def solve(N, K, A):
    # Create a set of all numbers from 1 to K
    all_numbers = set(range(1, K + 1))
    # Create a set of all numbers in A
    numbers_in_A = set(A)
    # Find the difference between the two sets
    missing_numbers = all_numbers - numbers_in_A
    # Return the sum of the missing numbers
    return sum(missing_numbers)

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Solve the problem and print the answer
print(solve(N, K, A))