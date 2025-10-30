from collections import Counter

s = input().strip()
count = Counter(s)
for i, char in enumerate(s):
    if count[char] == 1:
        print(i + 1)
        break