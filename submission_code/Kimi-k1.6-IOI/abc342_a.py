import sys
from collections import Counter

s = sys.stdin.readline().strip()
counter = Counter(s)
for char, count in counter.items():
    if count == 1:
        unique_char = char
        break
print(s.index(unique_char) + 1)