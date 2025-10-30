def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = []
    D = []
    for i in range(N):
        T.append(int(data[1 + 2*i]))
        D.append(int(data[2 + 2*i]))
    
    # Create a list of intervals (start, end)
    intervals = []
    for i in range(N):
        start = T[i]
        end = T[i] + D[i]
        intervals.append((start, end))
    
    # Sort intervals based on end time
    intervals.sort(key=lambda x: x[1])
    
    max_prints = 0
    last_print_time = -1
    
    for interval in intervals:
        start, end = interval
        # The earliest time we can print is max(start, last_print_time + 1)
        earliest_print = max(start, last_print_time + 1)
        if earliest_print <= end:
            max_prints += 1
            last_print_time = earliest_print
    
    print(max_prints)

if __name__ == "__main__":
    main()