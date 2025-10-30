# Read the input string
s = input().strip()

# Find the positions of the first and last '|' characters
first = s.find('|')
second = s.rfind('|')

# Combine the parts before the first and after the second '|'
result = s[:first] + s[second+1:]

# Print the result
print(result)