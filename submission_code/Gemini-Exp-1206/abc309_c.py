def solve():
    n, k = map(int, input().split())
    medicines = []
    for _ in range(n):
        medicines.append(list(map(int, input().split())))

    days = {}
    for a, b in medicines:
        if 1 not in days:
            days[1] = 0
        days[1] += b
        if a + 1 not in days:
            days[a + 1] = 0
        days[a + 1] -= b

    sorted_days = sorted(days.items())
    
    total_pills = 0
    last_day = 0
    for day, pill_change in sorted_days:
        total_pills += pill_change
        if total_pills <= k:
            print(day)
            return
        
solve()