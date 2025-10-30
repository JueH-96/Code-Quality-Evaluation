# Read the first line and split into parts
line1 = input().strip().split()
N = int(line1[0])
c1 = line1[1]
c2 = line1[2]

# Read the second line as the string S
S = input().strip()

# Initialize an empty string for the result
result = ''

# Iterate through each character in S
for char in S:
    if char == c1:
        result += char
    else:
        result += c2

# Print the result
print(result)