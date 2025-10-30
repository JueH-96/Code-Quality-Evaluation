def main():
    import sys
    input = sys.stdin.read
    # Read the input
    data = input().strip()
    if not data:
        return
    N = int(data)
    
    # Edge case: if N=1, then x=0 and y=0 works since 2^0*3^0 = 1, so print Yes.
    if N == 1:
        print("Yes")
        return
    
    # Remove factor of 2 as long as possible:
    while N % 2 == 0:
        N //= 2
    
    # Remove factor of 3 as long as possible:
    while N % 3 == 0:
        N //= 3
    
    # If N becomes 1, then only factors were 2 and 3.
    if N == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()