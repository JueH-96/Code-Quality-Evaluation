def last_substring(s):
    """
    This function prints the last substring when S is split by .s.
    In other words, it prints the longest suffix of S that does not contain ..
    """
    # Find the last occurrence of '.' in the string
    last_dot_index = s.rfind('.')

    # If '.' is not found, print the entire string
    if last_dot_index == -1:
        print(s)
    else:
        # Print the substring from the last '.' to the end of the string
        print(s[last_dot_index + 1:])

# Read input from stdin
s = input()

# Call the function
last_substring(s)