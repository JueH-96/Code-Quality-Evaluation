def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # Remove factors of 2
    while N % 2 == 0:
        N //= 2
    # Remove factors of 3
    while N % 3 == 0:
        N //= 3
        
    # If the resulting number is 1, then N was of the form 2^x * 3^y.
    print("Yes" if N == 1 else "No")

if __name__ == '__main__':
    main()