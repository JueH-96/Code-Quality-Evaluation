def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    ans = 1
    for i in range(n):
        count = i - a[i]
        ans = (ans * count) % mod
    print(ans)

solve()