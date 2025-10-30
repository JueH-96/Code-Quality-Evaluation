def remove_abc(s):
    """
    Removes all occurrences of 'ABC' from the string s.
    
    Args:
        s (str): The input string.
    
    Returns:
        str: The string after removing all occurrences of 'ABC'.
    """
    while 'ABC' in s:
        s = s.replace('ABC', '', 1)
    return s

# Read the input from stdin
s = input()

# Remove all occurrences of 'ABC' from the string
result = remove_abc(s)

# Print the result
print(result)