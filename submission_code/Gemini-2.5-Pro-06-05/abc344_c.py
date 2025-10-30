import sys

def solve():
    """
    Reads three sequences A, B, C and a list of query values X.
    For each query value x in X, determines if there exist a in A, b in B, c in C
    such that a + b + c = x.
    """
    # Read input sequences
    try:
        n_str = sys.stdin.readline()
        if not n_str: return
        N = int(n_str)
        A = list(map(int, sys.stdin.readline().split()))
        M = int(sys.stdin.readline())
        B = list(map(int, sys.stdin.readline().split()))
        L = int(sys.stdin.readline())
        C = list(map(int, sys.stdin.readline().split()))
        Q = int(sys.stdin.readline())
        X = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle empty input or malformed lines gracefully
        return

    # To optimize, we only need the unique values from each sequence.
    # Using sets is both Pythonic and efficient for this.
    unique_A = set(A)
    unique_B = set(B)
    unique_C = set(C)

    # Pre-compute all possible sums and store them in a set for fast O(1) lookups.
    # The number of possible sums is at most |unique_A|*|unique_B|*|unique_C|,
    # which is at most 100*100*100 = 10^6, a manageable number.
    # A set comprehension is a concise and efficient way to achieve this.
    possible_sums = {a + b + c for a in unique_A for b in unique_B for c in unique_C}

    # Process each query
    # For each query x, we check for its existence in the pre-computed set of sums.
    output = []
    for x in X:
        if x in possible_sums:
            output.append("Yes")
        else:
            output.append("No")
    
    print("
".join(output))

solve()