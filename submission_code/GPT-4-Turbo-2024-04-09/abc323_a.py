# YOUR CODE HERE
S = input().strip()

# Check if every even-indexed position (1-based) has '0'
# This means checking indices 1, 3, 5, ..., 15 in 0-based indexing
result = all(S[i] == '0' for i in range(1, 16, 2))

if result:
    print("Yes")
else:
    print("No")