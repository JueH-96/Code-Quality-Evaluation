# Read N from standard input
N = int(input())

# Generate the result string
# 'x' if kick number is divisible by 3 (fails), 'o' otherwise (succeeds)
result = ''.join('x' if i % 3 == 0 else 'o' for i in range(1, N+1))

# Print the result
print(result)