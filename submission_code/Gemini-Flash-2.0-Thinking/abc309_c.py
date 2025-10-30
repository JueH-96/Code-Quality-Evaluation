def solve():
    n, k = map(int, input().split())
    medicines = []
    for _ in range(n):
        medicines.append(list(map(int, input().split())))

    changes = {}
    for a, b in medicines:
        changes[1] = changes.get(1, 0) + b
        changes[a + 1] = changes.get(a + 1, 0) - b

    sorted_days = sorted(changes.keys())

    current_pills = 0
    for day in sorted_days:
        current_pills += changes[day]
        if current_pills <= k:
            print(day)
            return

solve()