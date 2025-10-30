def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # For each product i, the input gives T and D.
    # The product is in the printer's range from time T to T + D.
    # We can choose any printing time in [T, T+D].
    intervals = []
    for _ in range(n):
        t = int(next(it))
        d = int(next(it))
        intervals.append((t, t + d))
    
    # Sort the intervals by the ending time (T + D).
    intervals.sort(key=lambda interval: interval[1])
    
    # Greedy algorithm:
    # current_time represents the earliest time at which the printer is available.
    # For each interval, we try to schedule a printing time as early as possible.
    # We choose x = max(start, current_time). If x is within the interval (x <= end),
    # then we can print on that product and update current_time to x+1 (accounting for the charging time).
    count = 0
    current_time = 0  # initial available time
    for start, end in intervals:
        x = max(start, current_time)
        if x <= end:
            count += 1
            # Update current_time: after printing at time x, printer is ready at time x+1.
            current_time = x + 1
    
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()