import bisect
from collections import defaultdict

s = input().strip()
t = input().strip()

# Preprocess the positions of each character in T
pos_dict = defaultdict(list)
for idx, char in enumerate(t):
    pos_dict[char].append(idx)

result = []
current_pos = 0
for char in s:
    # Get the list of positions for the current character
    indices = pos_dict[char]
    # Find the first position >= current_pos using binary search
    i = bisect.bisect_left(indices, current_pos)
    selected = indices[i]
    result.append(selected + 1)  # Convert to 1-based index
    current_pos = selected + 1

print(' '.join(map(str, result)))