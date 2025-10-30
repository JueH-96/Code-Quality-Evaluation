from collections import Counter

def most_frequent_char(s):
    """
    Find the character that appears most frequently in a string.
    If multiple such characters exist, report the one that comes earliest in alphabetical order.

    Args:
        s (str): The input string.

    Returns:
        str: The most frequent character.
    """
    # Count the frequency of each character
    char_count = Counter(s)

    # Find the maximum frequency
    max_count = max(char_count.values())

    # Find all characters with the maximum frequency
    max_chars = [char for char, count in char_count.items() if count == max_count]

    # Return the character that comes earliest in alphabetical order
    return min(max_chars)

# Read the input from stdin
s = input()

# Find and print the most frequent character
print(most_frequent_char(s))