def solve():
    import sys
    N = int(sys.stdin.readline())
    
    # Count the trailing zeros in the binary representation of N.
    # Method: if N is positive, the trailing zero count is ( (N & -N).bit_length() - 1 ).
    # Explanation: (N & -N) isolates the rightmost set bit of N.
    # Its bit_length gives the position of that set bit (1-based),
    # so subtracting 1 yields the count of zeros to the right of it.
    
    trailing_zeros = (N & -N).bit_length() - 1
    print(trailing_zeros)

def main():
    solve()

if __name__ == "__main__":
    main()