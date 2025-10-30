# YOUR CODE HERE
S = input().strip()

# Check if all even positions (1-based: 2, 4, 6, 8, 10, 12, 14, 16) have '0'
# In 0-based indexing, these are positions 1, 3, 5, 7, 9, 11, 13, 15
all_even_positions_zero = True

for i in range(1, 16, 2):  # indices 1, 3, 5, 7, 9, 11, 13, 15
    if S[i] != '0':
        all_even_positions_zero = False
        break

if all_even_positions_zero:
    print("Yes")
else:
    print("No")