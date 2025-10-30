import sys

def solve():
    """
    This function solves a single test case.
    """
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, K = map(int, line1.split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # The operation A_i = A_j where |i-j| <= K allows value propagation.
    # We can model this with a graph where indices are vertices and an edge
    # (i, j) exists if |i-j| <= K.
    # A value at an index can be propagated to any other index in the same
    # connected component.

    # With the constraint 1 <= K < N, for any index i (from 0 to N-2),
    # i and i+1 are connected because |i - (i+1)| = 1 <= K.
    # This forms a path 0-1-2-...-(N-1), so all indices are in one
    # single connected component.

    # Therefore, the set of values that any A[i] can be changed to is the set
    # of all unique values initially present in the entire array A.
    # For A to be transformable into B, every value in B must be from this set.
    
    # Create a set of unique values from A for efficient lookup.
    available_values = set(A)
    
    # Check if all values required for B are available.
    for val in B:
        if val not in available_values:
            print("No")
            return
            
    print("Yes")

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        # Read the number of test cases.
        T_str = sys.stdin.readline()
        if not T_str:
            return
        T = int(T_str)
        # Process each test case.
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Gracefully handle empty input or malformed integers.
        pass

if __name__ == "__main__":
    main()