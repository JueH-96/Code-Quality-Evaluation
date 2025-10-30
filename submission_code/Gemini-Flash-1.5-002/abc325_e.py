# YOUR CODE HERE
import sys

def solve():
    line = sys.stdin.readline().split()
    n = int(line[0])
    a = int(line[1])
    b = int(line[2])
    c = int(line[3])

    d = []
    for _ in range(n):
        d.append(list(map(int, sys.stdin.readline().split())))

    inf = float('inf')
    dist_car = [[inf] * n for _ in range(n)]
    dist_train = [[inf] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dist_car[i][j] = d[i][j] * a
            dist_train[i][j] = d[i][j] * b + c

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist_car[i][j] = min(dist_car[i][j], dist_car[i][k] + dist_car[k][j])
                dist_train[i][j] = min(dist_train[i][j], dist_train[i][k] + dist_train[k][j])

    ans = inf
    for k in range(n):
        ans = min(ans, dist_car[0][k] + dist_train[k][n-1])
    ans = min(ans, dist_car[0][n-1])

    print(ans)

solve()