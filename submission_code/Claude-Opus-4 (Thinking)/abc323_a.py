# YOUR CODE HERE
S = input().strip()

# Check all even positions (2, 4, 6, 8, 10, 12, 14, 16)
# In 0-based indexing, these are positions 1, 3, 5, 7, 9, 11, 13, 15
if all(S[i-1] == '0' for i in range(2, 17, 2)):
    print("Yes")
else:
    print("No")