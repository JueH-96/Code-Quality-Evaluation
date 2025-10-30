def count_substrings(s):
    """
    Counts the number of different non-empty substrings in a given string.

    Args:
    s (str): The input string.

    Returns:
    int: The number of different non-empty substrings.
    """
    substrings = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substrings.add(s[i:j])
    return len(substrings)

# Read the input from stdin
s = input().strip()

# Count the number of different non-empty substrings
count = count_substrings(s)

# Print the result
print(count)