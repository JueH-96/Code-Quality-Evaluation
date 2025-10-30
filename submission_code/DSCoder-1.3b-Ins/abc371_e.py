import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    count = [0] * (N + 1)
    for i in range(N):
        count[A[i]] += 1

    result = 0
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            result += count[i] * count[j]

    print(result)

solve()