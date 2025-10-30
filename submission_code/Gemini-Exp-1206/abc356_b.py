def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    x = []
    for _ in range(n):
        x.append(list(map(int, input().split())))
    
    nutrient_totals = [0] * m
    for food in x:
        for i in range(m):
            nutrient_totals[i] += food[i]
    
    for i in range(m):
        if nutrient_totals[i] < a[i]:
            print("No")
            return
    
    print("Yes")

solve()