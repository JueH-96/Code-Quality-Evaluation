import sys
import itertools
import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_minimum_time(N, S, T, segments):
    min_time = float('inf')
    
    # Generate all permutations of the segments
    for perm in itertools.permutations(segments):
        # Try both directions for each segment in the permutation
        for directions in itertools.product([0, 1], repeat=N):
            current_time = 0.0
            current_position = (0, 0)
            
            for i in range(N):
                (A, B, C, D) = perm[i]
                if directions[i] == 0:
                    # Print from (A, B) to (C, D)
                    move_to_start = euclidean_distance(current_position[0], current_position[1], A, B) / S
                    print_segment = euclidean_distance(A, B, C, D) / T
                    current_position = (C, D)
                else:
                    # Print from (C, D) to (A, B)
                    move_to_start = euclidean_distance(current_position[0], current_position[1], C, D) / S
                    print_segment = euclidean_distance(C, D, A, B) / T
                    current_position = (A, B)
                
                current_time += move_to_start + print_segment
            
            min_time = min(min_time, current_time)
    
    return min_time

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    T = int(data[2])
    
    segments = []
    index = 3
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        D = int(data[index + 3])
        segments.append((A, B, C, D))
        index += 4
    
    result = calculate_minimum_time(N, S, T, segments)
    print(result)