def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append((l - 1, r - 1, x - 1))

    count = 0
    import itertools
    for p in itertools.permutations(range(n)):
        valid = True
        for l, r, x in conditions:
            max_val = -1
            for i in range(l, r + 1):
                max_val = max(max_val, p[i])
            if max_val == p[x]:
                valid = False
                break
        if valid:
            count += 1

    print(count % 998244353)

solve()