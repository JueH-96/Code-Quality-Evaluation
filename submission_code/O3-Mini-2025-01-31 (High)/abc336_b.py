def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # Read the integer value
    N = int(input_data[0])
    
    # Count the number of trailing zeros
    count = 0
    while N % 2 == 0:
        count += 1
        N //= 2
        
    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()