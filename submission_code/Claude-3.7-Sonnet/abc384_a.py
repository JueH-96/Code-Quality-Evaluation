# YOUR CODE HERE
# Read input
first_line = input().split()
c_1 = first_line[1]
c_2 = first_line[2]

S = input()

# Perform replacements
result = ""
for char in S:
    if char == c_1:
        result += c_1
    else:
        result += c_2

# Output result
print(result)