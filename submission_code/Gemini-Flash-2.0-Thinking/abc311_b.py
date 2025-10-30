def solve():
    n, d = map(int, input().split())
    schedules = [input() for _ in range(n)]

    max_consecutive_days = 0
    current_consecutive_days = 0

    for day in range(d):
        all_free_today = True
        for person in range(n):
            if schedules[person][day] == 'x':
                all_free_today = False
                break

        if all_free_today:
            current_consecutive_days += 1
        else:
            max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
            current_consecutive_days = 0

    max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
    print(max_consecutive_days)

solve()