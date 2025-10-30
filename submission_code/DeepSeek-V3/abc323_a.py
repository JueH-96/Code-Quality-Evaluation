# Read the input string
S = input().strip()

# Check each even position (2, 4, ..., 16)
# Note: Python uses 0-based indexing, so position i in the problem corresponds to index i-1 in Python
all_zero = True
for i in range(1, 16, 2):
    if S[i] != '0':
        all_zero = False
        break

# Print the result
if all_zero:
    print("Yes")
else:
    print("No")