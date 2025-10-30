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

    for start_time in queries:
        current_time = start_time + x
        for i in range(n - 1):
            wait_time = (p[i] - (current_time % p[i])) % p[i]
            current_time += wait_time + t[i]
        current_time += y
        print(current_time)

solve()