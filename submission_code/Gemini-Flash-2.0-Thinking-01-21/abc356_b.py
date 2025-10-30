def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    x = []
    for _ in range(n):
        x.append(list(map(int, input().split())))

    nutrient_totals = [0] * m
    for j in range(m):
        for i in range(n):
            nutrient_totals[j] += x[i][j]

    met_all_goals = True
    for j in range(m):
        if nutrient_totals[j] < a[j]:
            met_all_goals = False
            break

    if met_all_goals:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()