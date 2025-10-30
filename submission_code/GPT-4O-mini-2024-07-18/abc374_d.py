import sys
import math
from itertools import permutations

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def time_to_print_segment(start, end, T):
    return distance(start[0], start[1], end[0], end[1]) / T

def time_to_move(start, end, S):
    return distance(start[0], start[1], end[0], end[1]) / S

def total_time(segments, S, T):
    N = len(segments)
    min_time = float('inf')
    
    # Try all permutations of the segments
    for perm in permutations(range(N)):
        current_position = (0, 0)
        current_time = 0.0
        
        for idx in perm:
            segment = segments[idx]
            start = (segment[0], segment[1])
            end = (segment[2], segment[3])
            
            # Move to the start of the segment
            move_time = time_to_move(current_position, start, S)
            current_time += move_time
            
            # Print the segment
            print_time = time_to_print_segment(start, end, T)
            current_time += print_time
            
            # Update current position to the end of the segment
            current_position = end
        
        # Check if this permutation gives a better time
        min_time = min(min_time, current_time)
        
    return min_time

def main():
    input_data = sys.stdin.read().strip().splitlines()
    first_line = input_data[0].split()
    N = int(first_line[0])
    S = int(first_line[1])
    T = int(first_line[2])
    
    segments = []
    for i in range(1, N + 1):
        segments.append(tuple(map(int, input_data[i].split())))
    
    result = total_time(segments, S, T)
    print(f"{result:.12f}")

if __name__ == "__main__":
    main()