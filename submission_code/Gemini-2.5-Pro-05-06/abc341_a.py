# YOUR CODE HERE
# Read the input integer N
N = int(input())

# Construct the string:
# "10" repeated N times gives N ones and N zeros, ending in '0'.
# Appending "1" adds one more '1', resulting in N+1 ones and N zeros,
# and maintains alternation.
result_string = "10" * N + "1"

# Print the resulting string
print(result_string)