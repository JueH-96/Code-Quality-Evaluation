def is_subsequence(sub, main):
    i = 0
    for char in main:
        if i < len(sub) and char == sub[i]:
            i += 1
    return i == len(sub)

S = input().strip()
T = input().strip()

# Convert S to uppercase for comparison
S_upper = S.upper()

# Check if T is an airport code for S
if is_subsequence(T, S_upper) or (T[2] == 'X' and is_subsequence(T[:2], S_upper)):
    print("Yes")
else:
    print("No")