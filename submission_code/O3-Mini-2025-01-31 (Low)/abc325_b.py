def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    
    # Read N
    it = iter(input_data)
    N = int(next(it))
    
    # List of (W, X)
    bases = []
    for _ in range(N):
        w = int(next(it))
        x = int(next(it))
        bases.append((w, x))
    
    max_employees = 0
    # Check for every possible meeting starting hour in UTC (0..23)
    for T in range(24):
        current_sum = 0
        for w, x in bases:
            # Compute local starting time by adding offset
            local_time = (T + x) % 24
            # Meeting is from local_time to local_time+1.
            # It must be fully within [9,18] i.e. start time >=9 and finish time <=18 => start <= 17
            if 9 <= local_time <= 17:
                current_sum += w
        max_employees = max(max_employees, current_sum)
    
    sys.stdout.write(str(max_employees))
    
if __name__ == '__main__':
    main()