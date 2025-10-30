# YOUR CODE HERE
import sys
import math
from itertools import permutations

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def min_time_to_print(N, S, T, segments):
    min_time = float('inf')
    
    for perm in permutations(segments):
        time = 0
        current_x, current_y = 0, 0
        
        for (A, B, C, D) in perm:
            dist1 = distance(current_x, current_y, A, B)
            dist2 = distance(current_x, current_y, C, D)
            
            if dist1 / S < dist2 / S:
                time += dist1 / S
                time += distance(A, B, C, D) / T
                current_x, current_y = C, D
            else:
                time += dist2 / S
                time += distance(C, D, A, B) / T
                current_x, current_y = A, B
        
        min_time = min(min_time, time)
    
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
    
    result = min_time_to_print(N, S, T, segments)
    print(f"{result:.20f}")

if __name__ == "__main__":
    main()