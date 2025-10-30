# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string from standard input, finds the most frequent character
    (with alphabetical tie-breaking), and prints it to standard output.
    """
    
    # Read the input string. input() is sufficient for the given constraints.
    S = input()

    # Create a list to store frequencies of each lowercase English letter.
    # The list is indexed 0-25, corresponding to 'a'-'z'.
    counts = [0] * 26

    # Populate the frequency list by iterating through the input string.
    for char in S:
        # The index for a character is its ASCII value minus the ASCII value of 'a'.
        # For example, ord('c') - ord('a') gives 2.
        counts[ord(char) - ord('a')] += 1

    # Find the maximum frequency value in our list.
    max_count = max(counts)

    # Find the index of the first occurrence of this maximum value.
    # Because the `counts` list is implicitly ordered by the alphabet
    # ('a' at index 0, 'b' at 1, etc.), `list.index()` finds the
    # index corresponding to the alphabetically earliest character among
    # those with the highest frequency. This handles the tie-breaking rule.
    result_index = counts.index(max_count)

    # Convert the index back into its corresponding character.
    # For example, index 2 corresponds to chr(ord('a') + 2), which is 'c'.
    result_char = chr(ord('a') + result_index)

    # Print the final result to standard output.
    print(result_char)

solve()