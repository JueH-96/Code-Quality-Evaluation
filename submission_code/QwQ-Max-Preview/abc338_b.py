from collections import Counter

s = input().strip()
counts = Counter(s)
max_count = max(counts.values())

for c in 'abcdefghijklmnopqrstuvwxyz':
    if counts.get(c, 0) == max_count:
        print(c)
        break