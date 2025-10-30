def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        h, t = input().split()
        queries.append((h, int(t)))

    l = 1
    r = 2
    ans = 0

    for h, t in queries:
        if h == 'L':
            diff = abs(t - l)
            ans += min(diff, n - diff)
            l = t
        else:
            diff = abs(t - r)
            ans += min(diff, n - diff)
            r = t
    
    print(ans)

solve()