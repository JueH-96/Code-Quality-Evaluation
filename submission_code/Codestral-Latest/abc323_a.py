# YOUR CODE HERE
S = input().strip()

# Check if the i-th character of S is 0 for every even number i from 2 through 16
# Note: Python uses 0-based indexing, so even indices are 1, 3, 5, ..., 15
# We need to check indices 1 through 15 (2nd to 16th characters in 1-based indexing)

# Initialize a flag to True
valid = True

# Loop through the even indices (1-based) which are 1, 3, 5, ..., 15 in 0-based indexing
for i in range(1, 16, 2):
    if S[i] != '0':
        valid = False
        break

# Print the result
if valid:
    print("Yes")
else:
    print("No")