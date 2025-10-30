from collections import Counter

s = input().strip()
counts = Counter(s)
for idx, char in enumerate(s, 1):
    if counts[char] == 1:
        print(idx)
        break