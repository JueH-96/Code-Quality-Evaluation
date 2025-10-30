def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Check if N can be expressed as 2^x * 3^y
    # First, factor out all the 2s
    x = 0
    while N % 2 == 0:
        N //= 2
        x += 1
    
    # Then, factor out all the 3s
    y = 0
    while N % 3 == 0:
        N //= 3
        y += 1
    
    # If what remains is 1, then it's expressible as 2^x * 3^y
    if N == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()