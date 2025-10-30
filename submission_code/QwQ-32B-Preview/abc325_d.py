def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    products = []
    idx = 1
    for _ in range(N):
        T = int(data[idx])
        D = int(data[idx + 1])
        products.append((T, T + D))
        idx += 2
    # Sort products by their exit time
    products.sort(key=lambda x: x[1])
    
    last_printing_time = -1
    count = 0
    for entry, exit in products:
        earliest_print_time = max(last_printing_time + 1, entry)
        if earliest_print_time <= exit:
            count += 1
            last_printing_time = earliest_print_time
    print(count)

if __name__ == "__main__":
    main()