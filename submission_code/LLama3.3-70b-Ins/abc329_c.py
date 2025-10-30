import sys

def count_repetitions(S):
    """
    Count the number of non-empty substrings of S that are repetitions of one character.

    Args:
    S (str): The input string.

    Returns:
    int: The number of non-empty substrings of S that are repetitions of one character.
    """
    # Initialize a set to store unique substrings
    unique_substrings = set()

    # Iterate over the string
    for i in range(len(S)):
        for j in range(i + 1, len(S) + 1):
            # Extract the substring
            substring = S[i:j]

            # Check if the substring is a repetition of one character
            if len(set(substring)) == 1:
                unique_substrings.add(substring)

    # Return the number of unique substrings
    return len(unique_substrings)

# Read the input from stdin
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

# Solve the problem and write the answer to stdout
print(count_repetitions(S))