# Read input
X = int(input())

# Initialize product and n
product = 1
n = 1

# Multiply until product reaches or exceeds X
while product < X:
    product *= (n + 1)
    n += 1

# Print the result
print(n)