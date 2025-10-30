# YOUR CODE HERE
# Read the input string
S = input().strip()

# Check if all characters at even positions (2, 4, 6, ..., 16) are 0
if all(S[i] == '0' for i in range(1, 16, 2)):
    print("Yes")
else:
    print("No")