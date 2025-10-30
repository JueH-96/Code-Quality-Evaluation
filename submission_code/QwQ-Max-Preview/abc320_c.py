M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

from collections import defaultdict
from itertools import product

# Precompute positions for each digit in each reel
positions = [defaultdict(list), defaultdict(list), defaultdict(list)]
for i, s in enumerate([S1, S2, S3]):
    for idx, c in enumerate(s):
        positions[i][c].append(idx)

min_time = None

# Check each digit from 0 to 9
for d in map(str, range(10)):
    # Check if the digit is present in all three reels
    if not (positions[0][d] and positions[1][d] and positions[2][d]):
        continue
    
    # Get all possible positions for this digit in each reel
    reel1 = positions[0][d]
    reel2 = positions[1][d]
    reel3 = positions[2][d]
    
    current_min = float('inf')
    
    # Iterate over all combinations of positions
    for p1, p2, p3 in product(reel1, reel2, reel3):
        sorted_p = sorted([p1, p2, p3])
        x, y, z = sorted_p
        
        if x < y < z:
            T = z
        elif x == y < z:
            T = max(x + M, z)
        elif x < y and y == z:
            T = y + M
        elif x == y == z:
            T = x + 2 * M
        
        if T < current_min:
            current_min = T
    
    # Update the global minimum time
    if current_min < (min_time if min_time is not None else float('inf')):
        min_time = current_min

if min_time is not None:
    print(min_time)
else:
    print(-1)