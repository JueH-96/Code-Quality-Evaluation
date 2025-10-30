# Read the input N
N = int(input())

# Construct the string
# The pattern is 10 repeated N times, followed by a final 1.
# This uses N zeros and N+1 ones, alternating, starting and ending with 1.
result_string = "10" * N + "1"

# Print the result
print(result_string)