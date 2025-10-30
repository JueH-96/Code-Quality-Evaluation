# YOUR CODE HERE
W, B = map(int, input().split())
pattern = "wbwbwwbwbwbw"

# Repeat pattern enough times to cover all possible substrings
# Since we check positions 0-11 and need W+B characters from each start,
# we need at least 11 + W + B characters total
s = pattern * ((11 + W + B) // 12 + 2)

# Check all possible starting positions (0 to 11)
found = False
for start in range(12):
    substring = s[start:start + W + B]
    if substring.count('w') == W and substring.count('b') == B:
        found = True
        break

print("Yes" if found else "No")