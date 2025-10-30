# Read the input from stdin
N = int(input().strip())

# Initialize the result string
result = ""

# Iterate over the range 1 to N (inclusive)
for i in range(1, N + 1):
    # Check if i is a multiple of 3
    if i % 3 == 0:
        result += "x"
    else:
        result += "o"

# Print the result
print(result)