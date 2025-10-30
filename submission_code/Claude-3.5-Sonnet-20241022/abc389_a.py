# Read the input string
S = input().strip()

# Extract first and last characters (digits) and convert to integers
first_digit = int(S[0])
last_digit = int(S[2])

# Calculate product
result = first_digit * last_digit

# Output the result
print(result)