# Read the first line and split into parts
line1 = input().strip()
parts = line1.split()
N = int(parts[0])
c_1 = parts[1]
c_2 = parts[2]

# Read the second line as the string S
S = input().strip()

# Build the new string with replacements
new_string = ''.join([c if c == c_1 else c_2 for c in S])

# Print the resulting string
print(new_string)