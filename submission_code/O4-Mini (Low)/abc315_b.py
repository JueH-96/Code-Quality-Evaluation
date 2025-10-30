def main():
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    total_days = sum(D)
    # The middle day index in 1-based counting
    mid = (total_days + 1) // 2

    # Walk through the months to find where the mid-th day falls
    for month_index, days_in_month in enumerate(D, start=1):
        if mid <= days_in_month:
            # It's in this month
            print(month_index, mid)
            return
        else:
            mid -= days_in_month

if __name__ == "__main__":
    main()