# Read the input string from standard input
S = input().strip()

# Count the number of uppercase and lowercase letters
uppercase_count = sum(1 for c in S if c.isupper())
lowercase_count = len(S) - uppercase_count

# Convert the string based on the counts
if uppercase_count > lowercase_count:
    S = S.upper()
else:
    S = S.lower()

# Print the result to standard output
print(S)