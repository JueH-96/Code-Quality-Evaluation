def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Read M, D and then y, m, d from the input
    M = int(input_data[0])
    D = int(input_data[1])
    y = int(input_data[2])
    m = int(input_data[3])
    d = int(input_data[4])
    
    # Calculate next day
    d += 1
    if d > D:
        d = 1
        m += 1
        if m > M:
            m = 1
            y += 1

    # Output the result
    print(y, m, d)

if __name__ == '__main__':
    main()