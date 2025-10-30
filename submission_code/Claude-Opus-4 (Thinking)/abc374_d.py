import math
from itertools import permutations

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

N, S, T = map(int, input().split())
segments = []
for _ in range(N):
    A, B, C, D = map(int, input().split())
    segments.append(((A, B), (C, D)))

min_time = float('inf')

# Try all permutations of segments
for perm in permutations(range(N)):
    # Try all orientations (2^N possibilities)
    for orient in range(1 << N):
        time = 0.0
        pos = (0, 0)
        
        for i in range(N):
            seg_idx = perm[i]
            seg = segments[seg_idx]
            
            if (orient >> i) & 1:
                # Print in reverse order
                start = seg[1]
                end = seg[0]
            else:
                # Print in normal order
                start = seg[0]
                end = seg[1]
            
            # Move to start without laser
            time += distance(pos, start) / S
            
            # Print segment with laser
            time += distance(start, end) / T
            
            # Update position
            pos = end
        
        min_time = min(min_time, time)

print(min_time)