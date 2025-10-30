from collections import Counter

s = input().strip()
count = Counter(s)
freq_counts = list(count.values())

valid = True
max_freq = max(freq_counts) if freq_counts else 0
for i in range(1, max_freq + 1):
    cnt = freq_counts.count(i)
    if cnt not in {0, 2}:
        valid = False
        break

print("Yes" if valid else "No")