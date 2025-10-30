# Read the input string from stdin
S = input()

# Find the indices of the two '|' characters
start = S.index('|')
end = S.rindex('|')

# Extract the substring before the first '|' and after the second '|'
result = S[:start] + S[end+1:]

# Print the result
print(result)