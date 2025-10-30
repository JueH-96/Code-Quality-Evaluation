# Read the input from stdin
input_str = input()
bits = [int(x) for x in input_str.split()]

# Calculate the result
result = 0
for i in range(64):
    result += bits[i] * (2 ** i)

# Print the result
print(result)