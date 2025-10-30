# Read input from stdin
N = int(input())

# Generate the string with N zeros and N+1 ones, alternating starting with 1
result = ("10" * N) + "1"

# Print the result to stdout
print(result)