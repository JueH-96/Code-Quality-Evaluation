def is_extended_abc(s):
    """
    Checks if a given string is an Extended ABC string.

    Args:
    s (str): The input string.

    Returns:
    bool: True if the string is an Extended ABC string, False otherwise.
    """
    # Initialize variables to store the indices of the last 'A', 'B', and 'C' characters
    last_a = -1
    last_b = -1
    last_c = -1

    # Iterate over the string to find the last 'A', 'B', and 'C' characters
    for i, char in enumerate(s):
        if char == 'A':
            last_a = i
        elif char == 'B':
            last_b = i
        elif char == 'C':
            last_c = i

    # Check if the string is an Extended ABC string
    if last_a != -1 and last_b != -1 and last_c != -1:
        # Check if all 'A's come before all 'B's and all 'B's come before all 'C's
        if last_a < last_b and last_b < last_c:
            return True

    # If the string is not an Extended ABC string, check if it's an Extended A, B, or C string
    if all(char == 'A' for char in s) or all(char == 'B' for char in s) or all(char == 'C' for char in s):
        return True

    return False


# Read the input string from stdin
s = input()

# Check if the string is an Extended ABC string
if is_extended_abc(s):
    print("Yes")
else:
    print("No")