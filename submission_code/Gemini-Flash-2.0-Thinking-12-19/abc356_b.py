def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    x = []
    for _ in range(n):
        x.append(list(map(int, input().split())))

    nutrient_totals = [0] * m
    for i in range(n):
        for j in range(m):
            nutrient_totals[j] += x[i][j]

    for j in range(m):
        if nutrient_totals[j] < a[j]:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    solve()