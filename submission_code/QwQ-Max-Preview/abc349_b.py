from collections import Counter

s = input().strip()
freq = Counter(s)
counts = list(freq.values())
if not counts:  # This case shouldn't occur as per constraints
    print("No")
    exit()

max_count = max(counts)
count_to_chars = {}
for c in counts:
    count_to_chars[c] = count_to_chars.get(c, 0) + 1

valid = True
for i in range(1, max_count + 1):
    current = count_to_chars.get(i, 0)
    if current not in {0, 2}:
        valid = False
        break

print("Yes" if valid else "No")