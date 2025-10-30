def solve():
    n, a, b, c = map(int, input().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, input().split())))

    dist_car = [[(0) for _ in range(n)] for _ in range(n)]
    dist_train = [[(0) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dist_car[i][j] = d[i][j] * a
            dist_train[i][j] = d[i][j] * b + c

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist_car[i][j] = min(dist_car[i][j], dist_car[i][k] + dist_car[k][j])
                dist_train[i][j] = min(dist_train[i][j], dist_train[i][k] + dist_train[k][j])

    ans = dist_car[0][n - 1]
    for i in range(n):
        ans = min(ans, dist_car[0][i] + dist_train[i][n - 1])

    print(ans)

solve()