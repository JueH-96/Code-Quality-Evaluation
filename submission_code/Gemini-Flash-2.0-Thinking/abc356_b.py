def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    x = []
    for _ in range(n):
        x.append(list(map(int, input().split())))

    total_nutrients = [0] * m
    for j in range(m):
        for i in range(n):
            total_nutrients[j] += x[i][j]

    met_all_goals = True
    for i in range(m):
        if total_nutrients[i] < a[i]:
            met_all_goals = False
            break

    if met_all_goals:
        print("Yes")
    else:
        print("No")

solve()