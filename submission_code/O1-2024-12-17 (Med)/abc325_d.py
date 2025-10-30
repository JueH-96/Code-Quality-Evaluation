def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    intervals = []
    idx = 1
    for _ in range(N):
        T_i = int(input_data[idx]); D_i = int(input_data[idx+1])
        idx += 2
        intervals.append((T_i, T_i + D_i))
    
    # Sort intervals by their end time (second value)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    current = -1  # Earliest time we can next print
    
    for start, end in intervals:
        # We want to schedule the print as early as possible but not before 'current'
        t = max(current, start)
        if t <= end:
            count += 1
            current = t + 1  # Next available print time is 1 microsecond after

    print(count)

# Do not remove this call or you will receive zero points for this task.
main()