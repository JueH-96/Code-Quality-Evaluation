def main():
    import sys
    data = sys.stdin.read().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    total_days = sum(D)
    # The middle day index (1-based)
    target = (total_days + 1) // 2

    cum = 0
    for month_idx, days_in_month in enumerate(D, start=1):
        if cum + days_in_month >= target:
            # target-th day falls in this month
            day_in_month = target - cum
            print(month_idx, day_in_month)
            return
        cum += days_in_month

if __name__ == "__main__":
    main()