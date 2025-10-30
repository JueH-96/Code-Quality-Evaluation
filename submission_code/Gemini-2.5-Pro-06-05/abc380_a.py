# YOUR CODE HERE
import collections

def solve():
    """
    Reads a 6-digit number and checks if it contains
    exactly one '1', two '2's, and three '3's.
    """
    # Read the 6-digit number as a string from standard input.
    n_str = input()

    # The problem is equivalent to checking if the multiset of digits in the
    # input number is the same as the multiset of digits in "122333".
    # The `collections.Counter` class is the perfect tool for creating and
    # comparing multisets (frequency maps).

    # Create a Counter for the digits of the input number.
    input_counts = collections.Counter(n_str)

    # Create the target Counter representing the required digit frequencies.
    # This can be easily done from a string that has the desired digits.
    required_counts = collections.Counter("122333")

    # The `==` operator for Counter objects checks if two counters have the
    # same items and counts. This is exactly what we need.
    if input_counts == required_counts:
        print("Yes")
    else:
        print("No")

solve()