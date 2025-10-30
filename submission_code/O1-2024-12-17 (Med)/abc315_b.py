def main():
    import sys

    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    total_days = sum(D)  # total number of days
    mid_day = (total_days + 1) // 2  # the middle day index in the year

    cumulative = 0
    for i, days_in_month in enumerate(D, start=1):
        if mid_day <= cumulative + days_in_month:
            # The middle day is in this month
            day_in_month = mid_day - cumulative
            print(i, day_in_month)
            return
        cumulative += days_in_month

# Do not forget to call main()!
main()