def solve():
    n, x, y = map(int, input().split())
    p = []
    t = []
    for _ in range(n - 1):
        pi, ti = map(int, input().split())
        p.append(pi)
        t.append(ti)
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))

    for qi in queries:
        time = qi
        time += x
        for i in range(n - 1):
            time = (time + p[i] - 1) // p[i] * p[i] + t[i]
        time += y
        print(time)

solve()