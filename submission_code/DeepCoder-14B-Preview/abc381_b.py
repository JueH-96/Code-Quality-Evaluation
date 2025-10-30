import sys
from collections import Counter

s = input().strip()

if len(s) % 2 != 0:
    print("No")
    sys.exit()

for i in range(0, len(s), 2):
    if s[i] != s[i+1]:
        print("No")
        sys.exit()

cnt = Counter(s)
for c in cnt:
    if cnt[c] != 2:
        print("No")
        sys.exit()

print("Yes")