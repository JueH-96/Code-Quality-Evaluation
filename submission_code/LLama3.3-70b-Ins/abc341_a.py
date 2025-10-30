# Read the input from stdin
N = int(input())

# Initialize an empty string to store the result
result = ""

# Loop through the range of N
for i in range(N):
    # Append '0' and '1' to the result string
    result += '0'
    result += '1'

# Append one more '1' to the result string
result += '1'

# Print the result
print(result)