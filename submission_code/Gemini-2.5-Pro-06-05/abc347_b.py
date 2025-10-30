# YOUR CODE HERE
import sys

def solve():
    """
    This function reads a string from standard input, calculates the number
    of its distinct non-empty substrings, and prints the result to standard output.
    """
    # Read the input string from stdin.
    # .strip() is used to remove any trailing newline characters.
    s = sys.stdin.readline().strip()
    
    # Get the length of the string.
    n = len(s)
    
    # A set is used to store the substrings.
    # Sets automatically handle uniqueness, so any duplicate substring
    # added to the set will be ignored.
    distinct_substrings = set()
    
    # Generate all possible non-empty substrings.
    # The outer loop iterates through all possible start indices.
    for i in range(n):
        # The inner loop iterates through all possible end indices
        # for a given start index.
        # Python's slicing S[i:j+1] creates the substring from index i to j.
        for j in range(i, n):
            substring = s[i : j+1]
            distinct_substrings.add(substring)
            
    # The number of distinct substrings is the final size of the set.
    print(len(distinct_substrings))

solve()