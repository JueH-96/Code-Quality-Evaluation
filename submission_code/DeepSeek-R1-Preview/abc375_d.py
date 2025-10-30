from collections import defaultdict

s = input().strip()

# Dictionary to store positions for each character (1-based)
pos = defaultdict(list)
for idx, c in enumerate(s):
    pos[c].append(idx + 1)

total = 0

for c in pos:
    lst = pos[c]
    m = len(lst)
    if m < 2:
        continue  # Need at least two positions to form a pair
    
    # Calculate sum of (xj - xi) for all i < j
    current_sum = 0
    count = 0
    s_total = 0
    for x in lst:
        s_total += count * x - current_sum
        current_sum += x
        count += 1
    
    # Subtract the number of pairs (m choose 2)
    num_pairs = m * (m - 1) // 2
    contribution = s_total - num_pairs
    total += contribution

print(total)