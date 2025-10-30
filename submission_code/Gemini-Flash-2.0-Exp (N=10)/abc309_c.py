def solve():
    n, k = map(int, input().split())
    medicines = []
    for _ in range(n):
        medicines.append(list(map(int, input().split())))

    day = 1
    while True:
        total_pills = 0
        for a, b in medicines:
            if day <= a:
                total_pills += b
        if total_pills <= k:
            print(day)
            return
        day += 1

solve()