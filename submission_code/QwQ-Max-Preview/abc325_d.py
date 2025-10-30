def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    products = []
    idx = 1
    for _ in range(n):
        t = int(input[idx])
        d = int(input[idx + 1])
        idx += 2
        end = t + d
        products.append((end, t))
    
    # Sort by end (ascending), then by t (ascending)
    products.sort()
    
    count = 0
    last_print_time = -1  # Initialize to -infinity (since first print can be at 0)
    
    for end, t in products:
        earliest = max(t, last_print_time + 1)
        if earliest <= end:
            count += 1
            last_print_time = earliest
    
    print(count)

if __name__ == "__main__":
    main()