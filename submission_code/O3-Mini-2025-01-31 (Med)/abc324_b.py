def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # Factor out powers of 2
    while N % 2 == 0:
        N //= 2
        
    # Factor out powers of 3
    while N % 3 == 0:
        N //= 3
        
    # After removing all factors of 2 and 3,
    # if N becomes 1, it means it was of the form 2^x * 3^y.
    print("Yes" if N == 1 else "No")

if __name__ == "__main__":
    main()