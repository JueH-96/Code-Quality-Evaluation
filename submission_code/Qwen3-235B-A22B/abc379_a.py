# Read the input
n = int(input())

# Extract each digit
a = n // 100
b = (n // 10) % 10
c = n % 10

# Compute the two required numbers
first = b * 100 + c * 10 + a
second = c * 100 + a * 10 + b

# Print the result
print(f"{first} {second}")