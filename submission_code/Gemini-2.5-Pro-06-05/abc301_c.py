import sys
from collections import Counter

# Read the two input strings from standard input.
s = sys.stdin.readline().strip()
t = sys.stdin.readline().strip()

# Use Counter to get the frequency of each character (card).
# This handles the "rearrangement" cheat, as only counts matter.
count_s = Counter(s)
count_t = Counter(t)

# Define the set of "magic" characters that '@' can become.
magic_chars = set("atcoder")

# --- Step 1: Check non-magic characters ---
# Iterate through all lowercase letters.
for char_code in range(ord('a'), ord('z') + 1):
    char = chr(char_code)
    # If a character is not magic, its count must be identical in both rows.
    if char not in magic_chars:
        if count_s[char] != count_t[char]:
            # If counts differ, it's impossible to win.
            print("No")
            sys.exit()

# --- Step 2: Calculate deficits for magic characters ---
# s_needs: Total '@' cards S must use to match T's initial magic counts.
# t_needs: Total '@' cards T must use to match S's initial magic counts.
s_needs = 0
t_needs = 0
for char in magic_chars:
    if count_s[char] < count_t[char]:
        # S is short, so it needs to create this character.
        s_needs += count_t[char] - count_s[char]
    elif count_t[char] < count_s[char]:
        # T is short, so it needs to create this character.
        t_needs += count_s[char] - count_t[char]

# --- Step 3: Check if deficits can be covered by '@' cards ---
# Each string must have enough '@' cards to cover its own needs.
if count_s['@'] >= s_needs and count_t['@'] >= t_needs:
    # If both can cover their deficits, a win is possible.
    print("Yes")
else:
    # If either cannot, it's impossible.
    print("No")