import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    ans = 0
    j = 0
    for i in range(N):
        while j < M and A[i] >= B[j]:
            j += 1
        ans = max(ans, A[i])

    print(ans)

solve()