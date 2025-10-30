def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    products = []
    for _ in range(N):
        T = int(input[idx])
        D = int(input[idx + 1])
        idx += 2
        end = T + D
        products.append((T, end))
    
    # Sort by end time, then by start time (T)
    products.sort(key=lambda x: (x[1], x[0]))
    
    count = 0
    last_print_time = -10**19  # A sufficiently small initial value
    
    for t, end in products:
        earliest = max(t, last_print_time + 1)
        if earliest <= end:
            count += 1
            last_print_time = earliest
    
    print(count)

if __name__ == "__main__":
    main()