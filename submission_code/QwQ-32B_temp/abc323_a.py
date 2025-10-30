s = input().strip()

# Check all even positions (1-based) which are 0-based indices 1, 3, ..., 15
all_zero = True
for i in range(1, 16, 2):
    if s[i] != '0':
        all_zero = False
        break

print("Yes" if all_zero else "No")