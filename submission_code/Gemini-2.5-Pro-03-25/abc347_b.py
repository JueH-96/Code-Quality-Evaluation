# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string from standard input, calculates the number of 
    distinct non-empty substrings, and prints the result to standard output.
    """
    # Read the input string from standard input
    s = sys.stdin.readline().strip()
    n = len(s)
    
    # Use a set to store unique substrings. Sets automatically handle duplicates.
    unique_substrings = set()

    # Generate all possible non-empty substrings
    # Iterate through all possible starting positions of the substring
    for i in range(n):
        # Iterate through all possible ending positions of the substring
        # The end index j must be greater than or equal to the start index i
        for j in range(i, n):
            # Extract the substring S[i...j]
            # In Python slicing S[i:j+1] extracts characters from index i up to, but not including, j+1.
            substring = s[i : j + 1]
            # Add the extracted substring to the set
            unique_substrings.add(substring)

    # The size of the set gives the number of distinct substrings
    print(len(unique_substrings))

# Call the solve function to execute the logic when the script is run
solve()