def main():
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    total_days = sum(D)
    # The middle day index (1-based)
    middle = (total_days + 1) // 2

    # Find which month and which day
    cum = 0
    for i, days in enumerate(D, start=1):
        if cum + days >= middle:
            day_in_month = middle - cum
            print(i, day_in_month)
            return
        cum += days

if __name__ == "__main__":
    main()