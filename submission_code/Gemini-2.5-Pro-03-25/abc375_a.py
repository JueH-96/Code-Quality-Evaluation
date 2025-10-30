# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    """
    Reads input N and string S, finds the number of occurrences of the pattern '#.#'
    in S, and prints the count.
    """
    # Read the integer N (number of seats) from standard input.
    n = int(sys.stdin.readline())
    # Read the string S (seat states) from standard input and remove the trailing newline.
    s = sys.stdin.readline().strip()

    # Initialize a counter for the number of times the condition is met.
    count = 0
    # Define the target pattern string we are looking for: "#.#".
    # This corresponds to seat i ('#'), seat i+1 ('.'), seat i+2 ('#').
    target_pattern = "#.#"

    # The problem asks for the number of integers i such that 1 <= i <= N - 2
    # and seats i, i+1, i+2 match '#', '.', '#' respectively.
    # In terms of the 0-indexed string S, this means we are looking for indices i-1, i, i+1
    # such that S[i-1] == '#', S[i] == '.', and S[i+1] == '#'.

    # An equivalent way is to look for the substring "#.#" within S.
    # Let j be the starting index of a potential match in the string S.
    # The substring of length 3 starting at j is S[j:j+3].
    # This substring covers indices j, j+1, and j+2.
    # For the indices to be valid, j+2 must be less than or equal to n-1 (the last index).
    # This implies j <= n-3.
    # So, we need to iterate through all possible starting indices j from 0 to n-3.
    # The Python range function `range(n - 2)` generates integers 0, 1, ..., n-3.
    # This loop will naturally handle cases where n < 3, as the range will be empty.
    for j in range(n - 2):
        # Extract the substring of length 3 starting at index j.
        # In Python slicing s[start:end], the character at index 'end' is not included.
        # So, s[j : j+3] correctly extracts characters at indices j, j+1, and j+2.
        substring = s[j : j+3]

        # Check if the extracted substring matches the target pattern "#.#".
        if substring == target_pattern:
            # If it matches, it means we found a valid configuration.
            # This corresponds to the problem's seat index i = j + 1 satisfying the condition.
            # Increment the counter.
            count += 1

    # Print the final count to standard output.
    print(count)

# Standard boilerplate for competitive programming to ensure the function runs
# when the script is executed.
if __name__ == '__main__':
    solve()