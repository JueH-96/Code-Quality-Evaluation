# Read the input string
s = input().strip()

# Split the string into parts using '.' as the delimiter
parts = s.split('.')

# The last part is the answer
print(parts[-1])