import sys

def main():
    N = int(sys.stdin.readline())

    # N is a positive integer (1 <= N <= 10^9).
    # We need to find ctz(N), the number of trailing zeros in N's binary representation.

    # Method: Bit manipulation
    # The expression `N & (-N)` isolates the lowest set bit of N.
    # If N = X...X10...0 (k zeros), then N & (-N) will be 0...010...0 (k zeros).
    # This value is 2^k.
    # For example, if N = 20 (binary 10100), k=2.
    #   N & (-N) = 20 & (-20) = 4 (binary 100, which is 2^2).
    # If N = 5 (binary 101), k=0.
    #   N & (-N) = 5 & (-5) = 1 (binary 1, which is 2^0).
    
    # Since N is positive, N & (-N) will also be positive and a power of 2.
    # Let `lowest_set_bit_value = N & (-N)`. This value is 2^k.
    # The method `int.bit_length()` returns the number of bits required to represent
    # an integer in binary, excluding the sign and leading zeros.
    # For a power of 2, say `val = 2^k` (where k >= 0):
    #   - If k=0, val=1, val.bit_length()=1.
    #   - If k=1, val=2, val.bit_length()=2.
    #   - If k=j, val=2^j, val.bit_length()=j+1.
    # So, k = val.bit_length() - 1.
    
    if N == 0:
        # This case is excluded by constraints (1 <= N <= 10^9).
        # If N could be 0, ctz(0) is usually undefined or taken as a special value.
        # The formula (0 & -0).bit_length() - 1 would yield -1, which is not standard.
        # However, for this problem, N is always positive.
        pass # This path is not reachable under the problem constraints.

    lowest_set_bit_value = N & (-N)
    
    # Since N >= 1, lowest_set_bit_value is always a power of 2 (e.g., 1, 2, 4, ...).
    # Thus, lowest_set_bit_value.bit_length() is always >= 1.
    ctz_N = lowest_set_bit_value.bit_length() - 1
    
    print(ctz_N)

if __name__ == '__main__':
    main()