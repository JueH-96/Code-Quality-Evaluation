# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Reads a string S from standard input, determines if it's a "good string"
    based on the specified criteria, and prints "Yes" or "No" to standard output.
    """
    # Read input string from standard input
    s = sys.stdin.readline().strip()

    # 1. Count the occurrences of each character in the string S
    # collections.Counter is efficient for this task.
    # Example: S = "commencement" -> char_counts = {'c': 2, 'o': 1, 'm': 3, 'e': 3, 'n': 2, 't': 1}
    char_counts = Counter(s)

    # 2. Count the occurrences of each frequency count
    # We need to know how many characters appear exactly i times.
    # First, get the frequencies of each character present in the string.
    # Example: S = "commencement" -> frequencies are [2, 1, 3, 3, 2, 1] (counts of c, o, m, e, n, t)
    character_frequencies = char_counts.values()

    # Now, count how many characters share the same frequency.
    # Example: S = "commencement" -> freq_counts = {1: 2, 2: 2, 3: 2}
    # This means:
    # - 2 characters appear exactly 1 time (o, t)
    # - 2 characters appear exactly 2 times (c, n)
    # - 2 characters appear exactly 3 times (e, m)
    freq_counts = Counter(character_frequencies)

    # 3. Check the condition for a "good string"
    # The condition states: "For all integers i not less than 1: There are exactly
    # zero or exactly two different letters that appear exactly i times in S."
    #
    # The `freq_counts` dictionary stores pairs (frequency i, count C_i), where C_i is the
    # number of distinct characters appearing exactly i times, but importantly,
    # it only includes entries where C_i > 0.
    #
    # The condition means that if C_i is *not* 0, it *must* be 2.
    # Therefore, we just need to check if all the values in our `freq_counts` dictionary
    # (which represent the non-zero C_i values) are equal to 2.
    is_good = True
    # Iterate through the values of freq_counts. Each value represents the number of
    # distinct characters that share the same appearance frequency.
    for count_of_characters_with_same_freq in freq_counts.values():
        # If this number is not 2, the string does not satisfy the condition.
        if count_of_characters_with_same_freq != 2:
            is_good = False
            # No need to check further once a violation is found.
            break

    # Print the result to standard output
    if is_good:
        print("Yes")
    else:
        print("No")

# Execute the solve function when the script runs
solve()