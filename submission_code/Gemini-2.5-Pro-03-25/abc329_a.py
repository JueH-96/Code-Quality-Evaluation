# YOUR CODE HERE
import sys

def solve():
    # Read the input string from standard input
    # sys.stdin.readline().strip() reads a line and removes the trailing newline character
    s = sys.stdin.readline().strip()

    # The string.join(iterable) method returns a string which is the concatenation
    # of the strings in iterable. A separator string (in this case, ' ') is placed
    # between consecutive elements of the iterable.
    # When applied to a string 's', it iterates through the characters of 's'.
    result = ' '.join(s)

    # Print the resulting string to standard output
    print(result)

# Call the solve function to execute the logic
solve()
# YOUR CODE HERE