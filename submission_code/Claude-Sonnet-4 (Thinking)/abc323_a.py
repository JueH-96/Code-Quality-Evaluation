S = input().strip()

# Check even positions (1-indexed): 2, 4, 6, 8, 10, 12, 14, 16
# In 0-indexed: 1, 3, 5, 7, 9, 11, 13, 15
if all(S[i] == '0' for i in range(1, 16, 2)):
    print("Yes")
else:
    print("No")