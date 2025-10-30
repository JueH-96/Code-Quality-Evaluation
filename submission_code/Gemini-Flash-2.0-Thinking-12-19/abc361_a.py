def solve():
    n, k, x = map(int, input().split())
    a = list(map(int, input().split()))

    b = []
    for i in range(n):
        b.append(a[i])
        if i == k - 1:
            b.append(x)

    print(*b)

solve()