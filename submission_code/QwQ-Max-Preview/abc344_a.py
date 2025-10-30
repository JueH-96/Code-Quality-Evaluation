# Read input from standard input
S = input().strip()

# Find the indices of the two '|' characters
first = S.find('|')
second = S.find('|', first + 1)

# Concatenate the parts before the first and after the second
result = S[:first] + S[second+1:]

# Print the result
print(result)