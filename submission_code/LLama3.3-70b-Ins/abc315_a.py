def remove_vowels(S):
    """
    Removes all occurrences of 'a', 'e', 'i', 'o', 'u' from the given string S.
    
    Args:
        S (str): The input string.
    
    Returns:
        str: The resulting string after removing all vowels.
    """
    vowels = 'aeiou'
    result = ''
    
    # Iterate over each character in the string
    for char in S:
        # Check if the character is not a vowel
        if char not in vowels:
            # If it's not a vowel, add it to the result string
            result += char
    
    return result

# Read the input string from stdin
S = input()

# Remove vowels from the string and print the result
print(remove_vowels(S))