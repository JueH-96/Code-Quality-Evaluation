# YOUR CODE HERE
# Read the input
N = input().strip()

# Extract the digits
a = N[0]
b = N[1]
c = N[2]

# Form the two new integers
first_integer = b + c + a
second_integer = c + a + b

# Print the results
print(first_integer, second_integer)