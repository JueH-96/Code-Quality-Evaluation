# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0

# For each value, find all positions where it appears
from collections import defaultdict
positions = defaultdict(list)
for i in range(n):
    positions[a[i]].append(i)

# For each value that appears at least twice
for value, pos_list in positions.items():
    if len(pos_list) < 2:
        continue
    
    # For each pair of positions (i, k) where this value appears
    for i_idx in range(len(pos_list)):
        for k_idx in range(i_idx + 1, len(pos_list)):
            i = pos_list[i_idx]
            k = pos_list[k_idx]
            
            # Count positions j between i and k where A[j] != value
            for j in range(i + 1, k):
                if a[j] != value:
                    count += 1

print(count)