# Read the input as a list of integers
bits = list(map(int, input().split()))

# Initialize the result to 0
result = 0

# Iterate over each bit and calculate the sum
for i in range(len(bits)):
    result += bits[i] * (1 << i)

# Print the result
print(result)