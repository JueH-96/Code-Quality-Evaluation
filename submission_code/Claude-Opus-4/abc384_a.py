# YOUR CODE HERE
# Read the first line
first_line = input().split()
N = int(first_line[0])
c_1 = first_line[1]
c_2 = first_line[2]

# Read the string S
S = input()

# Build the result string
result = ""
for char in S:
    if char == c_1:
        result += char
    else:
        result += c_2

# Print the result
print(result)