def check_string(s):
    """
    Checks if the i-th character of S is 0 for every even number i from 2 through 16.

    Args:
        s (str): The input string of length 16 consisting of 0 and 1.

    Returns:
        str: "Yes" if the condition is met, "No" otherwise.
    """
    # Iterate over the even indices from 2 to 16 (inclusive)
    for i in range(2, 17, 2):
        # Check if the character at the current index is not '0'
        if s[i-1] != '0':  # Subtract 1 from i because indices are 0-based
            # If the character is not '0', return "No"
            return "No"
    # If the loop completes without finding any non-'0' characters, return "Yes"
    return "Yes"

# Read the input string from stdin
s = input()

# Call the function and print the result
print(check_string(s))