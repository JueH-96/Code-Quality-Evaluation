import sys

S = input()

# Split the string by '.'
parts = S.split('.')

# Get the last part (the longest suffix without '.')
result = parts[-1]

print(result)