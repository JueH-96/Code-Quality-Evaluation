import sys

def solve():
    """
    Reads input, solves the problem based on case analysis, and prints the output.
    """
    try:
        # Read N from stdin
        n_str = sys.stdin.readline()
        if not n_str:
            return
        N = int(n_str)
        
        # Read the sequence A from stdin
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle cases with malformed or empty input
        return

    # Count the number of zeros in the sequence A
    num_zeros = A.count(0)
    
    if num_zeros == 0:
        # Case 1: A contains only 1s.
        # No operations needed, so any string is good.
        print("Yes")
    elif num_zeros == N:
        # Case 2: A contains only 0s.
        # A solution exists if and only if N is even.
        if N % 2 == 0:
            print("Yes")
        else:
            print("No")
    else:
        # Case 3: A contains a mix of 0s and 1s.
        # A satisfying assignment for operations always exists.
        print("Yes")

if __name__ == "__main__":
    solve()