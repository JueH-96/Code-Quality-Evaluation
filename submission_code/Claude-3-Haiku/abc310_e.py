def nand(a, b):
    return 0 if a == b == 1 else 1

def f(i, j, A):
    if i == j:
        return A[i-1]
    return f(i, j-1, A) ^ A[j-1]

def solve(N, S):
    A = [int(c) for c in S]
    total = 0
    for i in range(1, N+1):
        for j in range(i, N+1):
            total += f(i, j, A)
    return total

# Read input
N = int(input())
S = input()

# Solve and print the answer
print(solve(N, S))