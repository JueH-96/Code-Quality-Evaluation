# Read the input from stdin
n = int(input())

# Extract the digits of the input number
a = n // 100
b = (n // 10) % 10
c = n % 10

# Construct the two output numbers
output1 = b * 100 + c * 10 + a
output2 = c * 100 + a * 10 + b

# Print the output to stdout
print(output1, output2)