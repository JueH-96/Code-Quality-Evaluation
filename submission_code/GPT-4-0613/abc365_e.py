def solve():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    ans = 0
    r = 0
    sum_A = [0]*(N+1)
    for i in range(N):
        while r < N and (sum_A[i] ^ (sum_A[r+1] ^ A[r])) == (sum_A[i] + A[r]):
            r += 1
            sum_A[r] = sum_A[r-1] + A[r-1]
        ans += r - i
        if i == r:
            r += 1
        else:
            sum_A[r] = sum_A[r-1] + A[r-1]
    print(ans)

solve()