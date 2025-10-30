from collections import Counter
import sys

s = sys.stdin.read().strip()
counts = Counter(s)

freq_of_freq = {}
for cnt in counts.values():
    if cnt in freq_of_freq:
        freq_of_freq[cnt] += 1
    else:
        freq_of_freq[cnt] = 1

max_count = max(freq_of_freq.keys()) if freq_of_freq else 0

for i in range(1, max_count + 1):
    current = freq_of_freq.get(i, 0)
    if current not in (0, 2):
        print("No")
        exit()

print("Yes")