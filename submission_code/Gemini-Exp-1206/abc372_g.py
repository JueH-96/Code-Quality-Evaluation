def solve():
    n = int(input())
    a = []
    b = []
    c = []
    for _ in range(n):
        ai, bi, ci = map(int, input().split())
        a.append(ai)
        b.append(bi)
        c.append(ci)

    ans = 0
    for x in range(1, 1001):
        for y in range(1, 1001):
            valid = True
            for i in range(n):
                if a[i] * x + b[i] * y >= c[i]:
                    valid = False
                    break
            if valid:
                ans += 1
    print(ans)

t = int(input())
for _ in range(t):
    solve()