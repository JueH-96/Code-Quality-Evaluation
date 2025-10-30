from collections import Counter
import sys

s = sys.stdin.readline().strip()

if len(s) % 2 != 0:
    print("No")
else:
    valid = True
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            valid = False
            break
    if not valid:
        print("No")
    else:
        cnt = Counter(s)
        for count in cnt.values():
            if count != 2:
                valid = False
                break
        print("Yes" if valid else "No")