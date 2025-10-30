import math

R = int(input())
threshold = R * R - 0.5

count = 0

# Maximum |i| such that i^2 + |i| <= threshold
max_abs_i = int((-1 + math.sqrt(1 + 4 * threshold)) / 2)

for abs_i in range(max_abs_i + 1):
    remaining = threshold - abs_i * abs_i - abs_i
    
    # Count j values such that j^2 + |j| <= remaining
    j_count = 1  # For j = 0
    
    # For |j| > 0
    if remaining >= 2:
        max_abs_j = int((-1 + math.sqrt(1 + 4 * remaining)) / 2)
        j_count += 2 * max_abs_j
    
    # Count i values for this |i|
    if abs_i == 0:
        count += j_count
    else:
        count += 2 * j_count

print(count)