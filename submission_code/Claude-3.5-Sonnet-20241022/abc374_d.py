import itertools
import math
import sys

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def solve():
    # Read input
    N, S, T = map(int, input().split())
    segments = []
    for _ in range(N):
        a, b, c, d = map(int, input().split())
        segments.append((a, b, c, d))
    
    # Try all possible permutations of segments and directions
    min_time = float('inf')
    
    for perm in itertools.permutations(range(N)):
        # Try all possible combinations of directions for each segment
        for directions in itertools.product([0, 1], repeat=N):
            time = 0
            curr_x, curr_y = 0, 0  # Start at origin
            
            # Process each segment in the current permutation
            for i, seg_idx in enumerate(perm):
                a, b, c, d = segments[seg_idx]
                
                # If direction is 1, swap endpoints
                if directions[i]:
                    a, c = c, a
                    b, d = d, b
                
                # Time to move to starting point of segment
                time += dist(curr_x, curr_y, a, b) / S
                
                # Time to print the segment
                time += dist(a, b, c, d) / T
                
                # Update current position
                curr_x, curr_y = c, d
            
            min_time = min(min_time, time)
    
    return min_time

# Print result with high precision
print(f"{solve()}")