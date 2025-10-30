def convert_string(s):
    """
    This function converts the input string according to the problem statement.
    
    If the number of uppercase letters in the string is greater than the number of lowercase letters, 
    it converts all lowercase letters to uppercase. Otherwise, it converts all uppercase letters to lowercase.
    
    Parameters:
    s (str): The input string.
    
    Returns:
    str: The converted string.
    """
    
    # Count the number of uppercase and lowercase letters in the string
    uppercase_count = sum(1 for char in s if char.isupper())
    lowercase_count = sum(1 for char in s if char.islower())
    
    # If the number of uppercase letters is greater than the number of lowercase letters, 
    # convert all lowercase letters to uppercase
    if uppercase_count > lowercase_count:
        return s.upper()
    
    # Otherwise, convert all uppercase letters to lowercase
    else:
        return s.lower()

# Read the input string from stdin
s = input()

# Convert the string and print the result
print(convert_string(s))