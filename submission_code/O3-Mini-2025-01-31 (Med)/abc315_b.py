def main():
    import sys
    input_data = sys.stdin.read().split()
    M = int(input_data[0])
    days = list(map(int, input_data[1:]))

    # Total number of days in the year
    total_days = sum(days)
    # Find the middle day. Since total_days is odd, (total_days+1)//2 is the middle day.
    middle_day = (total_days + 1) // 2

    cumulative = 0
    for month in range(M):
        cumulative += days[month]
        if cumulative >= middle_day:
            # The required day is in month (month+1).
            # Find the day in the month by subtracting the cumulative days of previous months.
            day_in_month = middle_day - (cumulative - days[month])
            print(month + 1, day_in_month)
            return

if __name__ == '__main__':
    main()