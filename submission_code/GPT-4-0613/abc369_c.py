def solve():
    N = int(input())
    A = list(map(int, input().split()))
    ans = 0
    l = 0
    while l < N:
        r = l
        while r+1 < N and A[r+1]-A[r] == A[l+1]-A[l]:
            r += 1
        ans += (r-l+1)*(r-l+2)//2
        l = r+1
    print(ans)

solve()