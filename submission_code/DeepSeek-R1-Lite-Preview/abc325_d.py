def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    products = []
    index = 1
    for _ in range(N):
        T = int(data[index])
        D = int(data[index + 1])
        products.append((T, T + D))
        index += 2
    
    # Sort products by end time
    products.sort(key=lambda x: x[1])
    
    last_print_time = -1
    count = 0
    
    for T_start, T_end in products:
        if T_start > last_print_time + 1:
            # Print at T_start
            last_print_time = T_start
            count += 1
        elif T_end > last_print_time + 1:
            # Print at last_print_time + 1
            last_print_time += 1
            count += 1
        # else: cannot print this product
    
    print(count)

if __name__ == "__main__":
    main()