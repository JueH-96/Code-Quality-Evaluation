# Read the input string S
S = input()

# Split the string S by the delimiter '|'
# Since S is guaranteed to contain exactly two '|'s,
# this will result in a list of three parts:
# parts[0]: substring before the first '|'
# parts[1]: substring between the two '|'s
# parts[2]: substring after the second '|'
parts = S.split('|')

# We need to remove the characters between the two '|'s,
# including the '|'s themselves. This means we keep
# parts[0] and parts[2].
result = parts[0] + parts[2]

# Print the resulting string
print(result)