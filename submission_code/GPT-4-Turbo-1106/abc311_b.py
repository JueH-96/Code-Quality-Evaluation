def max_consecutive_free_days(N, D, schedules):
    max_days = 0
    current_days = 0

    for day in range(D):
        if all(schedule[day] == 'o' for schedule in schedules):
            current_days += 1
            max_days = max(max_days, current_days)
        else:
            current_days = 0

    return max_days

# Read input from stdin
N, D = map(int, input().split())
schedules = [input().strip() for _ in range(N)]

# Calculate and print the result
print(max_consecutive_free_days(N, D, schedules))