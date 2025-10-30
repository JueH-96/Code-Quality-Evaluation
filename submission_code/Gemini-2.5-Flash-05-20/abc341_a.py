import sys

def solve():
    N = int(sys.stdin.readline())

    # The pattern is N pairs of '10' followed by a final '1'.
    # This ensures N zeros and N+1 ones, starting with '1' and alternating.
    # For example, if N=4:
    # "10" * 4 results in "10101010" (4 ones, 4 zeros)
    # Adding "1" at the end makes it "101010101" (5 ones, 4 zeros)
    result = "10" * N + "1"
    
    sys.stdout.write(result + "
")

if __name__ == '__main__':
    solve()