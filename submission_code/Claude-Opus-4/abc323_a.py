# YOUR CODE HERE
S = input().strip()

# Check all even positions (2, 4, 6, 8, 10, 12, 14, 16)
# In 0-based indexing, these are positions 1, 3, 5, 7, 9, 11, 13, 15
all_zeros = True
for i in range(1, 16, 2):  # 1, 3, 5, 7, 9, 11, 13, 15
    if S[i] != '0':
        all_zeros = False
        break

if all_zeros:
    print("Yes")
else:
    print("No")