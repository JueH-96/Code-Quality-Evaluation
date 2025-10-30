# YOUR CODE HERE
import sys

def solve():
    """
    This function reads the generated string S from standard input,
    reconstructs the sequence A, and prints it to standard output.
    """
    # Read the entire input line from stdin.
    s = sys.stdin.readline().strip()

    # The string S is constructed as: |<A_1 hyphens>|<A_2 hyphens>|...|<A_N hyphens>|
    # We can split the string by the '|' delimiter to isolate the groups of hyphens.
    # For example, "|---|-|----|" splits into ['', '---', '-', '----', ''].
    parts = s.split('|')

    # The first and last elements of the split list are empty because the string
    # starts and ends with the delimiter. The parts we are interested in are
    # the ones in the middle. We can get them by slicing the list.
    hyphen_groups = parts[1:-1]

    # The sequence A is composed of the lengths of these hyphen groups.
    # We can use a list comprehension to create the sequence A.
    a_sequence = [len(group) for group in hyphen_groups]

    # To print the result in the required format (A_1 A_2 ... A_N),
    # we can convert each integer in the list to a string and join them with spaces.
    # An alternative and more concise way is using the splat (*) operator with print.
    print(*a_sequence)

solve()