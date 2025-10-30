# Read the binary digits
A = list(map(int, input().split()))

# Calculate the decimal value
result = sum(A[i] * (2**i) for i in range(64))

# Print the result
print(result)