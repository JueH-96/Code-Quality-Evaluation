import sys

def main():
    """
    Reads a string from standard input, determines if it is an
    Extended ABC string, and prints the result to standard output.
    """
    # Read the single line of input and remove any leading/trailing whitespace.
    s = sys.stdin.readline().strip()
    
    # An "Extended ABC string" has a specific structure: all 'A's come first,
    # followed by all 'B's, followed by all 'C's. This is equivalent to the
    # property that the string is sorted alphabetically.
    # We can check this by comparing the string to its sorted version.
    
    # Create the sorted version of the string.
    # sorted(s) returns a sorted list of characters, e.g., ['A', 'B', 'C'].
    # "".join(...) concatenates them back into a string, e.g., "ABC".
    s_sorted = "".join(sorted(s))
    
    # Compare the original string with its sorted version.
    if s == s_sorted:
        # If they are identical, the string meets the criteria.
        print("Yes")
    else:
        # Otherwise, it does not.
        print("No")

if __name__ == "__main__":
    main()