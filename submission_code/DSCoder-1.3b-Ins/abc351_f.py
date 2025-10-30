import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    result = 0
    for i in range(N):
        for j in range(i+1, N):
            result += max(A[j] - A[i], 0)

    print(result)

solve()