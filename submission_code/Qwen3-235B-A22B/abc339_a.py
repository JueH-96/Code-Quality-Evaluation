# Read the input string
s = input().strip()

# Find the last occurrence of '.'
last_dot = s.rfind('.')

# Extract the substring after the last dot
result = s[last_dot + 1:]

# Print the result
print(result)