def solve():
    N, M, C, K = map(int, input().split())
    A = list(map(int, input().split()))
    A.sort()
    B = [0] * N
    for i in range(N):
        B[i] = (A[i] - i * C) % M
    B.sort()
    ans = 0
    for i in range(min(N, K)):
        ans += (B[i] + i * C) % M
    ans += max(0, K - N) * B[0]
    print(ans)

solve()