def main():
    import sys
    # Read the input integer X from standard input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    X = int(data[0])
    
    # Initialize the sum
    total = 0
    
    # Iterate through the 9x9 multiplication table
    for i in range(1, 10):
        for j in range(1, 10):
            value = i * j
            # Add the value to total if it does NOT equal X
            if value != X:
                total += value
                
    # Output the computed sum
    print(total)

if __name__ == '__main__':
    main()