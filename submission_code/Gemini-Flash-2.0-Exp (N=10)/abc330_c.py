def solve():
    d = int(input())
    ans = d
    x = 0
    while x * x <= d:
        y_sq = d - x * x
        y = int(y_sq**0.5)
        ans = min(ans, abs(x * x + y * y - d))
        y += 1
        ans = min(ans, abs(x * x + y * y - d))
        x += 1
    print(ans)

solve()