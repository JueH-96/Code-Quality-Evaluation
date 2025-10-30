# YOUR CODE HERE
# Read input values
N, c1, c2 = input().split()
N = int(N)
S = input().strip()

# Initialize the result string
result = []

# Iterate through each character in S
for char in S:
    if char == c1:
        result.append(char)
    else:
        result.append(c2)

# Convert the list to a string and print
print(''.join(result))