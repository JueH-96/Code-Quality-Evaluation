import sys

def solve(N, M, C, K, A):
    # Calculate the gcd of M and C
    gcd = euclidean(M, C)
    # Reduce M and C by their gcd
    M //= gcd
    C //= gcd
    # Reduce A by M
    A = [a % M for a in A]
    # Sort A
    A.sort()
    # Calculate the number of full cycles
    full_cycles = K // M
    # Calculate the remainder
    remainder = K % M
    # Initialize the result
    result = 0
    # Calculate the result for full cycles
    for i in range(M):
        result += A[i]
    result *= full_cycles
    # Calculate the result for the remainder
    for i in range(remainder):
        result += A[i]
    return result

def euclidean(a, b):
    if b == 0:
        return a
    return euclidean(b, a % b)

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
C = int(data[2])
K = int(data[3])
A = list(map(int, data[4:]))

# Solve the problem
result = solve(N, M, C, K, A)

# Print the result
print(result)