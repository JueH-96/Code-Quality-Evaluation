# YOUR CODE HERE
import sys

def solve():
    n, a, b, c = map(int, sys.stdin.readline().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, sys.stdin.readline().split())))

    inf = float('inf')
    dp_car = [[inf] * n for _ in range(n)]
    dp_train = [[inf] * n for _ in range(n)]

    for i in range(n):
        dp_car[i][i] = 0
        dp_train[i][i] = 0
    
    for start in range(n):
        for end in range(n):
            if start != end:
                dp_car[start][end] = d[start][end] * a

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dp_car[i][j] = min(dp_car[i][j], dp_car[i][k] + dp_car[k][j])

    for start in range(n):
        for end in range(n):
            if start != end:
                dp_train[start][end] = d[start][end] * b + c
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dp_train[i][j] = min(dp_train[i][j], dp_train[i][k] + dp_train[k][j])

    ans = inf
    for i in range(n):
        ans = min(ans, dp_car[0][i] + dp_train[i][n-1])

    print(ans)


solve()