def solve():
    n, x, y = map(int, input().split())
    pt = []
    for _ in range(n - 1):
        pt.append(list(map(int, input().split())))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))

    for start_time in queries:
        current_time = start_time + x
        for i in range(n - 1):
            p, t = pt[i]
            if current_time % p == 0:
                current_time += t
            else:
                current_time = ((current_time // p) + 1) * p + t
        current_time += y
        print(current_time)

solve()