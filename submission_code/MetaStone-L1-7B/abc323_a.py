S = input().strip()
required_indices = [1, 3, 5, 7, 9, 11, 13, 15]
all_zero = True

for idx in required_indices:
    if S[idx] != '0':
        all_zero = False
        break

print("Yes" if all_zero else "No")