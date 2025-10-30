# YOUR CODE HERE
import sys

def solve():
    """
    Reads N strings, sorts them by length, concatenates them, and prints the result.
    """
    # Read the number of strings
    n = int(sys.stdin.readline())

    # Read the N strings into a list
    strings = []
    for _ in range(n):
        # Read each string and strip potential trailing newline characters
        strings.append(sys.stdin.readline().strip())

    # Sort the list of strings based on their length in ascending order.
    # The `key=len` argument tells the sort function to use the length of each string
    # as the basis for comparison.
    strings.sort(key=len)

    # Concatenate the strings in the sorted order.
    # The `"".join(iterable)` method efficiently joins elements of an iterable (like our list)
    # into a single string, with an empty string "" as the separator.
    result = "".join(strings)

    # Print the final concatenated string to standard output.
    print(result)

# Call the solve function to execute the logic
solve()