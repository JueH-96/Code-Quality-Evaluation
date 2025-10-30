import sys

def is_1122_string(s):
    # Check if the length of the string is even
    if len(s) % 2 != 0:
        return "No"
    
    # Check if each pair of characters is equal
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            return "No"
    
    # Check if each character appears exactly twice
    char_count = {}
    for c in s:
        if c in char_count:
            char_count[c] += 1
        else:
            char_count[c] = 1
    
    for count in char_count.values():
        if count != 0 and count != 2:
            return "No"
    
    return "Yes"

# Read the input from stdin
s = input()

# Call the function and print the output to stdout
print(is_1122_string(s))