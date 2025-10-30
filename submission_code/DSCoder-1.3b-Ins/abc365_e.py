import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    result = 0
    for i in range(N-1):
        for j in range(i+1, N):
            result += A[i] ^ A[j]

    print(result)

solve()