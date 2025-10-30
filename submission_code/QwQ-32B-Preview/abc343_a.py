# Read input
input_str = input()

# Split the input into A and B
parts = input_str.split()
A = int(parts[0])
B = int(parts[1])

# Calculate A + B
sum_ab = A + B

# Calculate (A + B + 1) % 10
result = (sum_ab + 1) % 10

# Print the result
print(result)