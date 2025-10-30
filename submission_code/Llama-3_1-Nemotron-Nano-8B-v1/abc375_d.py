from collections import defaultdict

S = input().strip()
index_map = defaultdict(list)

for idx, char in enumerate(S):
    index_map[char].append(idx + 1)  # Using 1-based index

total = 0

for chars in index_map.values():
    m = len(chars)
    if m < 2:
        continue
    # Compute prefix sums
    prefix = [0] * m
    prefix[0] = chars[0]
    for i in range(1, m):
        prefix[i] = prefix[i-1] + chars[i]
    # Calculate contribution for each j
    for j in range(1, m):
        sum_ai = prefix[j-1]
        current = chars[j] * j - sum_ai - j
        total += current

print(total)