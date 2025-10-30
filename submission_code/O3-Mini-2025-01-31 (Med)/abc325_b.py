def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    bases = []
    index = 1
    for i in range(N):
        # Read W (number of employees) and X (local time offset at 0:00 UTC)
        W = int(data[index])
        X = int(data[index + 1])
        bases.append((W, X))
        index += 2

    max_employees = 0

    # The meeting must start at a whole hour in UTC (0 <= t < 24)
    # For each base, an employee can attend if the meeting (which lasts for 1 hour)
    # is completely within the local time window 9:00 to 18:00. That means,
    # the meeting start time (local) must be between 9 and 17 (inclusive) as it ends by 18.
    for t in range(24):
        total = 0
        for (W, X) in bases:
            # Calculate the local meeting start for this base.
            local_time = (t + X) % 24
            if 9 <= local_time <= 17:
                total += W
        if total > max_employees:
            max_employees = total

    print(max_employees)

if __name__ == "__main__":
    main()