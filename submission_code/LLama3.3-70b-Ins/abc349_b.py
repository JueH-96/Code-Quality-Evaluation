from collections import Counter

def is_good_string(s):
    """
    Determine if a string is a good string.

    A string S is a good string if and only if it satisfies the following property for all integers i not less than 1:
    - There are exactly zero or exactly two different letters that appear exactly i times in S.

    Args:
        s (str): The input string.

    Returns:
        bool: True if the string is a good string, False otherwise.
    """
    # Count the frequency of each character in the string
    char_count = Counter(s)

    # Initialize a dictionary to store the count of characters that appear i times
    count_dict = {}

    # Populate the count dictionary
    for char, count in char_count.items():
        if count not in count_dict:
            count_dict[count] = 1
        else:
            count_dict[count] += 1

    # Check if the string is a good string
    for count in count_dict.values():
        if count not in [0, 2]:
            return False

    return True

# Read the input string from stdin
s = input()

# Determine if the string is a good string
if is_good_string(s):
    print("Yes")
else:
    print("No")