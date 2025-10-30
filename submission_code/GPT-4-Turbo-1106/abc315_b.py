def find_middle_day(M, days):
    total_days = sum(days)
    middle_day = (total_days + 1) // 2
    current_day = 0
    for i, d in enumerate(days):
        current_day += d
        if current_day >= middle_day:
            month = i + 1
            day_of_month = d - (current_day - middle_day)
            return month, day_of_month
    return None

# Read input from stdin
M = int(input().strip())
days = list(map(int, input().strip().split()))

# Find the middle day
month, day_of_month = find_middle_day(M, days)

# Write output to stdout
print(f"{month} {day_of_month}")