S = input().strip()
n = len(S)
res = 0
current_pos = 0  # 0-based in the desired pattern (0: expect 'i', 1: expect 'o', etc.)

i = 0
while i < n:
    expected_char = 'i' if current_pos % 2 == 0 else 'o'
    if S[i] == expected_char:
        current_pos += 1
        i += 1
    else:
        # Need to insert expected_char before S[i]
        res += 1
        current_pos += 1
        # i remains the same
# After processing all characters, check if current_pos is even (desired length is even)
remaining = 0
if current_pos % 2 != 0:
    remaining = 1
    res += 1
print(res)