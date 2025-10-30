# YOUR CODE HERE
# Read the integer N from Standard Input
N = int(input())

# Convert the integer N to its string representation. This is the "digit N".
# For example, if N is 3, digit_N_as_string will be "3".
digit_N_as_string = str(N)

# The number of times to concatenate this digit is N itself.
# Python's string multiplication operator (*) concatenates a string with itself a specified number of times.
# For example, "3" * 3 results in "333".
result_string = digit_N_as_string * N

# Print the resulting string to Standard Output.
print(result_string)