# Read the input string
S = input().strip()

# Split the string by '.' and get the last part
parts = S.split('.')
last_part = parts[-1]

# Print the last part
print(last_part)