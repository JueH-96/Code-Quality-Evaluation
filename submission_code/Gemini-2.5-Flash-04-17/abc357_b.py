import sys

# Read the input string from standard input
s = sys.stdin.readline().strip()

# Count the number of uppercase and lowercase letters
uppercase_count = 0
lowercase_count = 0

for char in s:
    if char.isupper():
        uppercase_count += 1
    elif char.islower():
        lowercase_count += 1
    # According to the problem constraints, the string only contains English letters,
    # so we don't need to handle other characters.

# Determine whether to convert to uppercase or lowercase
# If the number of uppercase letters is greater than the number of lowercase letters,
# convert the entire string to uppercase.
if uppercase_count > lowercase_count:
    print(s.upper())
# Otherwise (if the number of uppercase letters is less than or equal to the number of lowercase letters),
# convert the entire string to lowercase.
else:
    print(s.lower())