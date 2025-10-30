def solve():
    n, k, q = map(int, input().split())
    a = [0] * n

    for _ in range(q):
        x, y = map(int, input().split())
        a[x - 1] = y
        b = sorted(a, reverse=True)
        f_a = sum(b[:k])
        print(f_a)

solve()