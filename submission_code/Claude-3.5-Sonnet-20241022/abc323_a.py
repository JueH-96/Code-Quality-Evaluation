S = input().strip()
is_valid = True

for i in range(1, 16, 2):  # Check even positions (1-based indexing)
    if S[i] != '0':
        is_valid = False
        break

print('Yes' if is_valid else 'No')