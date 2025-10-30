def solve():
    n, d = map(int, input().split())
    schedules = [input() for _ in range(n)]
    is_all_free = [False] * d
    for day_index in range(d):
        all_free_today = True
        for person_index in range(n):
            if schedules[person_index][day_index] == 'x':
                all_free_today = False
                break
        if all_free_today:
            is_all_free[day_index] = True
            
    max_consecutive_days = 0
    current_consecutive_days = 0
    for day_index in range(d):
        if is_all_free[day_index]:
            current_consecutive_days += 1
        else:
            max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
            current_consecutive_days = 0
            
    max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
    print(max_consecutive_days)

if __name__ == '__main__':
    solve()