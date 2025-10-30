# YOUR CODE HERE
import sys

def solve():
    """
    Reads strings S and T from stdin, determines if there exists a pair
    of integers (c, w) satisfying the given conditions, and prints "Yes" or "No".
    """
    # Read S and T from standard input, separated by a space
    s, t = sys.stdin.readline().split()
    n = len(s)  # Length of string S

    # Iterate through all possible values for the split width 'w'
    # According to the problem statement, 1 <= w < |S|
    # So, w ranges from 1 to n-1 (inclusive).
    for w in range(1, n):
        # Iterate through all possible values for the character position 'c'
        # According to the problem statement, 1 <= c <= w
        # So, c ranges from 1 to w (inclusive).
        for c in range(1, w + 1):
            # Use a list to efficiently build the constructed string
            constructed_chars = []

            # Iterate through the starting indices of the conceptual substrings
            # The substrings conceptually start at index 0, w, 2*w, and so on,
            # as long as the starting index is less than n.
            for i in range(0, n, w):
                # Calculate the 0-based index in the original string S
                # that corresponds to the c-th character (1-based) of the
                # conceptual substring starting at index i.
                # The 0-based index within the substring is c-1.
                # The absolute index in S is i + (c-1).
                target_index = i + c - 1

                # Check if this calculated index is valid (within the bounds of S).
                # If target_index < n, the character S[target_index] exists.
                # This condition implicitly ensures that the conceptual substring
                # starting at index `i` has a length of at least `c`.
                # If the character at index `i + c - 1` exists, the substring
                # `S[i: min(i+w, n)]` must contain at least `c` characters.
                if target_index < n:
                    # If the index is valid, append the character at that index
                    # to our list of characters.
                    constructed_chars.append(s[target_index])

            # Join the collected characters to form the candidate string
            constructed_t = "".join(constructed_chars)

            # Compare the constructed string with the target string T
            if constructed_t == t:
                # If they match, we have found a valid pair (c, w)
                # Print "Yes" and exit the function immediately.
                print("Yes")
                return

    # If the nested loops complete without finding any valid (c, w) pair
    # that produces the string T, then no such pair exists.
    # Print "No".
    print("No")

# Call the solve function to execute the logic
solve()