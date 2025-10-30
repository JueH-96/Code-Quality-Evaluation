from itertools import accumulate

def solve(A):
    N = len(A)
    B = list(accumulate(A, max))
    C = list(accumulate(A[::-1], max))[::-1]
    ans = 0
    for i in range(N):
        k = B[i]
        ok = -1
        ng = N
        while ng - ok > 1:
            m = (ok + ng) // 2
            if C[m] >= k:
                ok = m
            else:
                ng = m
        if i <= ok:
            ans = max(ans, (i + 1) + (N - ok - 1))
        else:
            ans = max(ans, (i) + (N - ok))
    return (ans + 1) // 2

N = int(input())
A = list(map(int, input().split()))
print(solve(A))