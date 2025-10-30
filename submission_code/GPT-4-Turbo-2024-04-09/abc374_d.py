import sys
import math
from itertools import permutations

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    T = int(data[2])
    
    segments = []
    index = 3
    for _ in range(N):
        A = int(data[index])
        B = int(data[index+1])
        C = int(data[index+2])
        D = int(data[index+3])
        segments.append(((A, B), (C, D)))
        index += 4
    
    # Calculate the minimal time to print all segments
    min_time = float('inf')
    
    # Try all permutations of the order of printing segments
    for perm in permutations(segments):
        # Try both directions for each segment
        for directions in itertools.product([0, 1], repeat=N):
            # Start at (0, 0)
            current_x, current_y = 0, 0
            total_time = 0.0
            
            for i, segment in enumerate(perm):
                (x1, y1), (x2, y2) = segment
                if directions[i] == 1:
                    (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
                
                # Move to the start of the segment
                move_time = distance(current_x, current_y, x1, y1) / S
                # Print the segment
                print_time = distance(x1, y1, x2, y2) / T
                
                total_time += move_time + print_time
                
                # Update current position to the end of the segment
                current_x, current_y = x2, y2
            
            # Update minimum time
            min_time = min(min_time, total_time)
    
    # Print the result with high precision
    print(f"{min_time:.20f}")