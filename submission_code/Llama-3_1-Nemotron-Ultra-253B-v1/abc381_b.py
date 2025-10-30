from collections import Counter

s = input().strip()

if len(s) % 2 != 0:
    print("No")
    exit()

for i in range(0, len(s), 2):
    if s[i] != s[i+1]:
        print("No")
        exit()

counts = Counter(s)

for cnt in counts.values():
    if cnt != 2:
        print("No")
        exit()

print("Yes")