# YOUR CODE HERE
W, B = map(int, input().split())

# Base pattern
pattern = "wbwbwwbwbwbw"

# Create a long enough string by repeating the pattern
# We need at least W+B characters, but let's use more to ensure we don't miss any valid substrings
# Using 2*(W+B) + len(pattern) to be safe
s = pattern * ((2*(W+B) + len(pattern)) // len(pattern) + 2)

# Check all substrings of length W+B
found = False
for start in range(len(s) - (W+B) + 1):
    substring = s[start:start + W + B]
    if substring.count('w') == W and substring.count('b') == B:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")