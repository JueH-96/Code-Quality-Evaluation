def solve():
    n, a, b = map(int, input().split())
    d = list(map(int, input().split()))

    for start_day in range(a + b):
        all_holidays = True
        for days_later in d:
            day_of_week = (start_day + days_later) % (a + b)
            if day_of_week == 0:
                day_of_week = a + b
            if day_of_week > a:
                all_holidays = False
                break
        if all_holidays:
            print("Yes")
            return

    print("No")

solve()