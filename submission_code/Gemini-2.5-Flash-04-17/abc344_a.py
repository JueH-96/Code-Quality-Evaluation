# YOUR CODE HERE
import sys

# Read the input string from stdin
S = sys.stdin.readline().strip()

# Find the index of the first occurrence of '|'
first_bar_index = S.find('|')

# Find the index of the second occurrence of '|', starting the search after the first '|'
second_bar_index = S.find('|', first_bar_index + 1)

# Extract the part of the string before the first '|'
part_before = S[:first_bar_index]

# Extract the part of the string after the second '|'
part_after = S[second_bar_index + 1:]

# Concatenate the two parts to get the resulting string
result = part_before + part_after

# Print the result to stdout
print(result)