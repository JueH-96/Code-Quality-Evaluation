# YOUR CODE HERE
import sys

# Read the input string from standard input
s = sys.stdin.readline().strip()

# Initialize counters for uppercase and lowercase letters
uppercase_count = 0
lowercase_count = 0

# Iterate through each character in the string
for char in s:
    # Check if the character is uppercase using the isupper() method
    if char.isupper():
        uppercase_count += 1
    # Check if the character is lowercase using the islower() method
    # Since the problem guarantees the string consists only of English letters,
    # if a character is not uppercase, it must be lowercase.
    # However, using elif char.islower() is explicit.
    elif char.islower():
        lowercase_count += 1

# Compare the counts of uppercase and lowercase letters
if uppercase_count > lowercase_count:
    # If the number of uppercase letters is greater,
    # convert the entire string to uppercase using the upper() method.
    result_string = s.upper()
else:
    # Otherwise (the number of lowercase letters is greater or equal,
    # but since the length is odd, it must be strictly greater),
    # convert the entire string to lowercase using the lower() method.
    result_string = s.lower()

# Print the resulting string to standard output
print(result_string)