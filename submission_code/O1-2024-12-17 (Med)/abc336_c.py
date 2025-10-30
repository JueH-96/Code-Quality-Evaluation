def main():
    import sys
    N = int(sys.stdin.readline().strip())
    
    # If N=1, the smallest good integer is 0
    if N == 1:
        print(0)
        return
    
    # Convert (N-1) to base-5
    x = N - 1
    base5_digits = []
    while x > 0:
        base5_digits.append(x % 5)
        x //= 5
    base5_digits.reverse()
    
    # Map each base-5 digit {0,1,2,3,4} to an even digit {0,2,4,6,8}
    mapping = "02468"
    result = ''.join(mapping[d] for d in base5_digits)
    
    print(result)

# Do not forget to call `main`!
if __name__ == "__main__":
    main()