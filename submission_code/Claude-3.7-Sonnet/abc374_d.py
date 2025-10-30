import math
from itertools import permutations

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve(N, S, T, lines):
    min_time = float('inf')
    
    # Try all permutations of line segments
    for perm in permutations(range(N)):
        # Try all possible starting endpoints for each line segment
        for mask in range(1 << N):
            curr_pos = (0, 0)
            total_time = 0
            
            for idx, line_idx in enumerate(perm):
                A, B, C, D = lines[line_idx]
                
                if (mask >> idx) & 1:  # Start from (C, D)
                    start, end = (C, D), (A, B)
                else:  # Start from (A, B)
                    start, end = (A, B), (C, D)
                
                # Time to move to the starting point
                total_time += distance(curr_pos, start) / S
                
                # Time to print the line
                total_time += distance(start, end) / T
                
                curr_pos = end
            
            min_time = min(min_time, total_time)
    
    return min_time

def main():
    N, S, T = map(int, input().split())
    lines = []
    for _ in range(N):
        lines.append(list(map(int, input().split())))
    
    result = solve(N, S, T, lines)
    print(result)

if __name__ == "__main__":
    main()