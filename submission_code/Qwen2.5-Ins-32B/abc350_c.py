import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    operations = []

    for i in range(N):
        if A[i] != i + 1:
            j = A.index(i + 1)
            A[i], A[j] = A[j], A[i]
            operations.append((i + 1, j + 1))

    print(len(operations))
    for op in operations:
        print(*op)

if __name__ == "__main__":
    solve()