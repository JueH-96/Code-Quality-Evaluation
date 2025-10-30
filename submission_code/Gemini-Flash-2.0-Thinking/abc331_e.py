def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    forbidden = set()
    for _ in range(l):
        u, v = map(int, input().split())
        forbidden.add((u, v))

    max_price = 0
    for i in range(n):
        for j in range(m):
            if (i + 1, j + 1) not in forbidden:
                max_price = max(max_price, a[i] + b[j])

    print(max_price)

solve()