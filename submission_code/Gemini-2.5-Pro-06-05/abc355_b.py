import sys

def solve():
    """
    Reads input sequences A and B, determines if the sorted combined sequence
    contains two consecutive elements from A, and prints the result.
    """
    # Read and discard the first line containing N and M.
    # The lengths of the lists are implicitly handled by Python.
    sys.stdin.readline()

    # Read the sequence A from the second line of standard input.
    A = list(map(int, sys.stdin.readline().split()))

    # Read the sequence B from the third line of standard input.
    B = list(map(int, sys.stdin.readline().split()))

    # Create the sequence C by combining A and B, then sorting it.
    C = sorted(A + B)

    # Convert list A to a set for efficient O(1) average time lookups.
    A_set = set(A)

    # Iterate through the sorted list C to check for adjacent pairs.
    # The loop runs up to the second-to-last element.
    for i in range(len(C) - 1):
        # Check if both the current element and the next one are in A.
        if C[i] in A_set and C[i+1] in A_set:
            # A consecutive pair from A has been found.
            print("Yes")
            # Terminate the script as the condition is met.
            return

    # If the loop completes, no such pair was found.
    print("No")

# Execute the solution.
solve()