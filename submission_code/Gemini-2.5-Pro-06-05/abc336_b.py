import sys

def solve():
    """
    Reads an integer N from standard input and prints the number of trailing
    zeros in its binary representation.
    """
    try:
        # Read the integer N from a single line of standard input.
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases where input is not a valid integer.
        # Given the problem constraints, this is not expected to happen.
        return

    # The number of trailing zeros (ctz) is the number of times N is divisible by 2.
    # We can find this by repeatedly checking the least significant bit (LSB)
    # and right-shifting the number until it becomes odd.
    
    # The problem constraints guarantee 1 <= N, so N is positive.
    # A defensive check for N > 0 is good practice but redundant here.
    
    count = 0
    # The loop continues as long as N is even.
    # (N & 1) == 0 is a fast way to check for evenness.
    while (N & 1) == 0:
        count += 1
        # N >>= 1 is a fast way to perform integer division by 2.
        N >>= 1
        
    print(count)

solve()