def solve():
    n, q = map(int, input().split())
    queries = list(map(int, input().split()))

    a = [0] * n
    s = set()

    for x in queries:
        if x in s:
            s.remove(x)
        else:
            s.add(x)

        size_s = len(s)
        for j in range(1, n + 1):
            if j in s:
                a[j - 1] += size_s

    print(*a)

solve()