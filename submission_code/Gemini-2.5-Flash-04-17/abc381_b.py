import sys
from collections import Counter

# Read the input string
S = sys.stdin.readline().strip()

# Condition 1: Length is even
if len(S) % 2 != 0:
    print("No")
    sys.exit()

# Condition 2: Characters at 2i and 2i+1 are equal (0-based indexing)
# For each i from 0 to |S|/2 - 1, check S[2*i] == S[2*i+1]
for i in range(len(S) // 2):
    if S[2 * i] != S[2 * i + 1]:
        print("No")
        sys.exit()

# Condition 3: Each character appears exactly zero or two times
char_counts = Counter(S)
# Iterate through the counts of characters that are present in S
for count in char_counts.values():
    # If any character has a count other than 2
    if count != 2:
        print("No")
        sys.exit()

# If all conditions passed
print("Yes")