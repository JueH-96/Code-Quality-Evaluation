# Read the input line from stdin
# The input consists of two integers A and B separated by a space
# input().split() reads the line and splits it into a list of strings
# map(int, ...) applies the int() function to each string in the list, converting them to integers
# The two integers are then unpacked into variables a and b
a, b = map(int, input().split())

# Calculate the sum of A and B
sum_of_a_and_b = a + b

# Calculate the square of the sum
# The ** operator is used for exponentiation
result = sum_of_a_and_b ** 2

# Print the final answer to stdout
print(result)