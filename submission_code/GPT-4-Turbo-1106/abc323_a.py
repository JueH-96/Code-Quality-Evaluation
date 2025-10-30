# Read the input string from standard input
S = input().strip()

# Check if every even-positioned character is '0'
is_valid = all(S[i] == '0' for i in range(1, 16, 2))

# Print the result
print("Yes" if is_valid else "No")