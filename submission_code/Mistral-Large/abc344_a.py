import sys

input = sys.stdin.read
data = input().strip()

# Find the positions of the two '|' characters
first_bar = data.index('|')
second_bar = data.index('|', first_bar + 1)

# Remove the characters between the two '|'s, including the '|'s themselves
result = data[:first_bar] + data[second_bar + 1:]

# Print the resulting string
print(result)