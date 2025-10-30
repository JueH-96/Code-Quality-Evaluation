import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    A.sort()

    ans = 0
    for i in range(N):
        x = A[i]
        l = r = i
        while l < N and A[l] < x + M // 2:
            l += 1
        while r < N and A[r] <= x + M:
            r += 1
        ans = max(ans, r - l)

    print(ans)

solve()