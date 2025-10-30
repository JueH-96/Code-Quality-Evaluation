import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    A.sort()
    total = 0
    for i in range(N):
        if i % 2 == 0:
            total += A[i]
        else:
            total -= A[i]

    print(total)

solve()