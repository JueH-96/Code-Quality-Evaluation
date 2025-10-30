def solve():
    n = int(input())
    a = []
    b = []
    x = []
    for _ in range(n - 1):
        ai, bi, xi = map(int, input().split())
        a.append(ai)
        b.append(bi)
        x.append(xi - 1)

    dist = [float('inf')] * n
    dist[0] = 0

    for i in range(n):
        if dist[i] == float('inf'):
            continue
        if i < n - 1:
            dist[i+1] = min(dist[i+1], dist[i] + a[i])
            dist[x[i]] = min(dist[x[i]], dist[i] + b[i])
            
    print(dist[n-1])

solve()