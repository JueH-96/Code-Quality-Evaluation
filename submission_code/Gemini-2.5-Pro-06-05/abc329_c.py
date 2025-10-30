# YOUR CODE HERE
import sys
from itertools import groupby

def solve():
    """
    This function solves the problem by finding the longest run for each character
    and summing these lengths.
    """
    # Read N (the length of the string). It's part of the input format but not
    # strictly required for this particular algorithm.
    try:
        N = int(sys.stdin.readline())
        # Read the string S.
        S = sys.stdin.readline().strip()
    except (ValueError, IndexError):
        # Handle cases where input might be empty. Given the problem constraints (N>=1),
        # this is just for robustness.
        print(0)
        return

    # A dictionary to store the maximum length of a contiguous run for each character.
    # For example, for S = "aaabaa", this will become {'a': 3, 'b': 1}.
    max_lengths = {}

    # `itertools.groupby` is an efficient tool for grouping consecutive identical
    # elements in an iterable.
    for char, group in groupby(S):
        # `group` is an iterator. We calculate its length to find the length
        # of the current run of `char`. Using sum with a generator expression
        # is memory-efficient as it avoids creating a temporary list.
        length = sum(1 for _ in group)
        
        # We need to find the longest run for each character.
        # We update the dictionary with the maximum length found so far for `char`.
        # The `.get(char, 0)` method safely retrieves the current max, defaulting
        # to 0 if `char` hasn't been seen before.
        max_lengths[char] = max(max_lengths.get(char, 0), length)

    # The total number of unique substrings is the sum of these maximum lengths.
    # If the longest run of 'a' is 3, we can form "a", "aa", "aaa" (3 unique substrings).
    # Summing these counts for all characters gives the final answer.
    total_count = sum(max_lengths.values())

    # Print the final result to standard output.
    print(total_count)

# Execute the solution.
solve()