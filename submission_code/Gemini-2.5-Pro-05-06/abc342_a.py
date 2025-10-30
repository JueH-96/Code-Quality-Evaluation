import collections

# Read the input string from Standard Input
S = input()

# Count the frequency of each character in S
# For example, if S = "yay", counts will be Counter({'y': 2, 'a': 1})
counts = collections.Counter(S)

# Find the character that appears only once
unique_char = ''
# According to the problem constraints, there will be exactly two distinct characters.
# One will have a count of 1, and the other will have a count of len(S) - 1.
# We iterate through the items in the Counter object (character and its count).
for char_val, count_val in counts.items():
    if count_val == 1:
        unique_char = char_val
        break  # Stop once the unique character is found

# Find the 0-indexed position of the unique character in the original string S.
# S.find(unique_char) returns the lowest index in S where unique_char is found.
# Since unique_char appears only once, this will be its actual index.
position_0_indexed = S.find(unique_char)

# The problem asks for the 1-indexed position, so add 1 to the 0-indexed position.
print(position_0_indexed + 1)