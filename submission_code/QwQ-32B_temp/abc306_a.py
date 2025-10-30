# Read the input values
n = int(input())
s = input().strip()

# Initialize the result string
result = ''

# Iterate through each character in the string and double it
for c in s:
    result += c * 2

# Print the resulting string
print(result)