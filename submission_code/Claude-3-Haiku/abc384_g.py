import sys

# Read the input
N = int(sys.stdin.readline().strip())
A = [int(x) for x in sys.stdin.readline().strip().split()]
B = [int(x) for x in sys.stdin.readline().strip().split()]
K = int(sys.stdin.readline().strip())
queries = [[int(y) for y in sys.stdin.readline().strip().split()] for _ in range(K)]

# Solve the problem
for Xk, Yk in queries:
    total = 0
    for i in range(Xk):
        for j in range(Yk):
            total += abs(A[i] - B[j])
    print(total)