def find_unique_char(S):
    """
    This function finds the index of the character in the string S that differs from all other characters.

    Args:
        S (str): The input string.

    Returns:
        int: The index of the unique character (1-indexed).
    """
    # Count the frequency of each character in the string
    char_count = {}
    for char in S:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the character that appears only once
    unique_char = [char for char, count in char_count.items() if count == 1][0]

    # Find the index of the unique character (1-indexed)
    index = S.index(unique_char) + 1

    return index

# Read the input from stdin
S = input()

# Find and print the index of the unique character
print(find_unique_char(S))