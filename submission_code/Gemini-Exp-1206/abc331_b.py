def solve():
    n, s, m, l = map(int, input().split())
    ans = float('inf')
    for i in range(101):
        for j in range(101):
            for k in range(101):
                if 6 * i + 8 * j + 12 * k >= n:
                    ans = min(ans, s * i + m * j + l * k)
    print(ans)

solve()