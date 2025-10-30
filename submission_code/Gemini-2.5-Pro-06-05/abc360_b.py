# YOUR CODE HERE
import sys

def solve():
    """
    Reads strings S and T from standard input and determines if there exists
    a pair of integers (c, w) satisfying the given conditions.
    """
    try:
        # Read the two strings from a single line of input.
        S, T = input().split()
    except (ValueError, EOFError):
        # Gracefully handle empty or malformed input.
        return

    len_S = len(S)

    # We need to find if there exists a pair (c, w) such that:
    # 1. 1 <= c <= w
    # 2. w < |S|

    # Iterate through all possible values for the split width 'w'.
    # The range for w is from 1 up to, but not including, the length of S.
    for w in range(1, len_S):
        # For each 'w', iterate through all possible character positions 'c'.
        # The range for c is from 1 up to w (inclusive).
        for c in range(1, w + 1):
            
            # Construct the candidate string according to the problem rules.
            # The c-th character corresponds to a 0-based index of c-1.
            # The required characters are at indices c-1, c-1+w, c-1+2w, etc.
            # Python's string slicing with a step, S[start::step], provides a
            # concise and efficient way to extract these characters.
            # This automatically handles the condition that we only consider
            # substrings of length at least 'c', because the character at
            # index `k*w + c-1` can only be accessed if the k-th substring
            # is long enough to contain it.
            
            candidate_string = S[c - 1::w]
            
            # Check if the generated string matches the target string T.
            if candidate_string == T:
                # If a match is found, we have satisfied the condition.
                print("Yes")
                # We can terminate the program immediately.
                return

    # If the loops complete without finding any valid (c, w) pair,
    # it means no such pair exists.
    print("No")

# Run the solution
solve()