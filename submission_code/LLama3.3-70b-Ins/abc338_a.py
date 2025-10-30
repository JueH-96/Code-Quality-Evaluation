def check_string(s):
    """
    Checks if the first character of a string is uppercase and all other characters are lowercase.

    Args:
        s (str): The input string.

    Returns:
        str: "Yes" if the condition is satisfied, "No" otherwise.
    """
    # Check if the first character is uppercase
    first_uppercase = s[0].isupper()
    
    # Check if all other characters are lowercase
    all_lowercase = s[1:].islower()
    
    # Return "Yes" if both conditions are met, "No" otherwise
    if first_uppercase and all_lowercase:
        return "Yes"
    else:
        return "No"

# Read the input string from stdin
s = input()

# Call the function and print the result
print(check_string(s))