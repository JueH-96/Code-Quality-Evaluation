import math
from itertools import permutations, product

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    line = input().split()
    N, S, T = int(line[0]), int(line[1]), int(line[2])
    
    segments = []
    for _ in range(N):
        a, b, c, d = map(int, input().split())
        segments.append(((a, b), (c, d)))
    
    min_time = float('inf')
    
    # Try all permutations of segments
    for perm in permutations(range(N)):
        # Try all combinations of endpoint choices (0 or 1 for each segment)
        for choices in product([0, 1], repeat=N):
            total_time = 0
            current_pos = (0, 0)
            
            for i, seg_idx in enumerate(perm):
                seg = segments[seg_idx]
                choice = choices[i]
                
                # Choose which endpoint to start from
                if choice == 0:
                    start_point = seg[0]
                    end_point = seg[1]
                else:
                    start_point = seg[1]
                    end_point = seg[0]
                
                # Time to move to start point (without laser)
                move_time = distance(current_pos, start_point) / S
                total_time += move_time
                
                # Time to print the segment (with laser)
                print_time = distance(start_point, end_point) / T
                total_time += print_time
                
                # Update current position
                current_pos = end_point
            
            min_time = min(min_time, total_time)
    
    print(f"{min_time:.20f}")

solve()