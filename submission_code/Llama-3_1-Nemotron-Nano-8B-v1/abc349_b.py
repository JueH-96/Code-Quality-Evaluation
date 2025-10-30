import sys
from collections import Counter

s = input().strip()
freq = Counter(s)
count_freq = Counter(freq.values())

max_count = max(freq.values()) if freq else 0

for i in range(1, max_count + 1):
    current = count_freq.get(i, 0)
    if current not in (0, 2):
        print("No")
        sys.exit()

print("Yes")