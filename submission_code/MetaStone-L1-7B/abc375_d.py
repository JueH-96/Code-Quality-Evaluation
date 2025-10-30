s = input().strip()
from collections import defaultdict

char_indices = defaultdict(list)

for i, c in enumerate(s):
    char_indices[c].append(i + 1)  # Positions are 1-based

total = 0

for c in char_indices:
    pos = char_indices[c]
    m = len(pos)
    if m < 2:
        continue
    sum_pairs = 0
    current_prefix = 0
    for j in range(m):
        sum_pairs += pos[j] * j - current_prefix
        current_prefix += pos[j]
    num_pairs = m * (m - 1) // 2
    contribution = sum_pairs - num_pairs
    total += contribution

print(total)