import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    operations = []
    for i in range(N):
        if A[i] != i + 1:
            j = A.index(i + 1)
            A[i], A[j] = A[j], A[i]
            operations.append((i + 1, j + 1))

    print(len(operations))
    for op in operations:
        print(op[0], op[1])

solve()