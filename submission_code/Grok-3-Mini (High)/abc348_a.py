# Read input from stdin
N = int(input())

# Generate the result string using list comprehension
result = ''.join(['x' if i % 3 == 0 else 'o' for i in range(1, N + 1)])

# Print the result to stdout
print(result)