from itertools import permutations
import math

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
    for orientation in range(1 << N):
        current_pos = (0, 0)
        total_time = 0
        
        for i, seg_idx in enumerate(perm):
            segment = segments[seg_idx]
            
            # Check orientation bit
            if (orientation >> i) & 1:
                # Start from first endpoint, go to second
                start, end = segment[0], segment[1]
            else:
                # Start from second endpoint, go to first
                start, end = segment[1], segment[0]
            
            # Move to start position (without laser)
            move_time = distance(current_pos, start) / S
            total_time += move_time
            
            # Print the segment (with laser)
            print_time = distance(start, end) / T
            total_time += print_time
            
            # Update current position
            current_pos = end
        
        min_time = min(min_time, total_time)

print(min_time)