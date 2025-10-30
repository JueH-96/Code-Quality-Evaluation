import sys

def vertical_text(strings):
    # Find the maximum length of the strings
    max_length = max(len(s) for s in strings)
    
    # Initialize the result as a list of empty strings
    result = [''] * max_length
    
    # Iterate over the strings in reverse order
    for s in reversed(strings):
        # Iterate over the characters in the string
        for i, c in enumerate(s):
            # Append the character to the corresponding position in the result
            result[i] += c
    
    # Append * to the end of each string in the result
    result = [s + '*' for s in result]
    
    return result

# Read the number of strings
n = int(sys.stdin.readline().strip())

# Read the strings
strings = [sys.stdin.readline().strip() for _ in range(n)]

# Solve the problem
result = vertical_text(strings)

# Print the result
for s in result:
    print(s.rstrip('*'))