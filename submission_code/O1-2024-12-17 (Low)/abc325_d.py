def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    intervals = []
    idx = 1
    for _ in range(N):
        T = int(input_data[idx]); D = int(input_data[idx+1])
        intervals.append((T, T + D))
        idx += 2
    
    # Sort by the ending times (T_i + D_i)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_print_time = -1  # something smaller than any valid time
    for start, end in intervals:
        # We want the earliest time >= last_print_time+1 that is within [start, end]
        candidate = max(last_print_time + 1, start)
        if candidate <= end:
            count += 1
            last_print_time = candidate
    
    print(count)

# Do not forget to call main() 
if __name__ == "__main__":
    main()