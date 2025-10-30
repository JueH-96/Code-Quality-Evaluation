def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for i in range(1, n):
        if a[i] == a[i-1] or b[i] == b[i-1]:
            raise ValueError("Input sequences are not good")

    for i in range(1, n):
        if a[i-1] == b[i] and b[i-1] == b[i]:
            print(-1)
            return

    ans = 0
    for i in range(n):
        diff_add = (b[i] - a[i] + m) % m
        diff_sub = (a[i] - b[i] + m) % m
        ans += min(diff_add, diff_sub)

    print(ans)

solve()