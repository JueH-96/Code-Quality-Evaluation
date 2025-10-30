def solve():
    n, d = map(int, input().split())
    schedules = [input() for _ in range(n)]
    is_free_day = [True] * d
    for day_index in range(d):
        for person_index in range(n):
            if schedules[person_index][day_index] == 'x':
                is_free_day[day_index] = False
                break
    
    max_consecutive_days = 0
    current_consecutive_days = 0
    for day_free in is_free_day:
        if day_free:
            current_consecutive_days += 1
        else:
            max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
            current_consecutive_days = 0
            
    max_consecutive_days = max(max_consecutive_days, current_consecutive_days)
    print(max_consecutive_days)

if __name__ == '__main__':
    solve()