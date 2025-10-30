def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    intervals = []
    idx = 1
    for _ in range(N):
        T = int(data[idx])
        D = int(data[idx + 1])
        # Each product can be printed at any time in [T, T + D].
        intervals.append((T, T + D))
        idx += 2
    
    # Sort intervals by their end time.
    intervals.sort(key=lambda x: x[1])
    
    answer = 0
    last_print_time = -10**30  # A very small number.
    
    for start, end in intervals:
        # We can print this product if we can find a time >= last_print_time + 1 in [start, end].
        print_time = max(start, last_print_time + 1)
        if print_time <= end:
            # We can print here.
            answer += 1
            last_print_time = print_time
    
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()