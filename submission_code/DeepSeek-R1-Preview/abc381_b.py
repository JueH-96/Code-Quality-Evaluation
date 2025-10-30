import sys
from collections import Counter

S = sys.stdin.readline().strip()

# Check if the length is even
if len(S) % 2 != 0:
    print("No")
    sys.exit()

# Check each consecutive pair
for i in range(0, len(S), 2):
    if S[i] != S[i+1]:
        print("No")
        sys.exit()

# Check each character appears exactly twice
counts = Counter(S)
for cnt in counts.values():
    if cnt != 2:
        print("No")
        sys.exit()

print("Yes")