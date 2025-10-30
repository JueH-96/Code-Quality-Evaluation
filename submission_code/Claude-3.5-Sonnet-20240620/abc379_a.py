# YOUR CODE HERE
# Read the input
N = input().strip()

# Extract digits
a, b, c = N

# Create the two required numbers
num1 = int(b + c + a)
num2 = int(c + a + b)

# Print the result
print(f"{num1} {num2}")