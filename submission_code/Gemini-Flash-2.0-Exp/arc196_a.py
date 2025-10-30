def solve():
    n = int(input())
    a = list(map(int, input().split()))

    if n % 2 != 0:
        a = a[:-1]
        n -= 1

    ans = 0
    for i in range(0, n, 2):
        ans += abs(a[i] - a[i+1])
    print(ans)

solve()