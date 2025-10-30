import sys
import math
from itertools import permutations

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def line_length(x1, y1, x2, y2):
    return dist(x1, y1, x2, y2)

def solve(N, S, T, lines):
    best_time = float('inf')
    
    for perm in permutations(range(N)):
        for directions in range(1 << N):
            time = 0
            x, y = 0, 0
            
            for i in range(N):
                line = lines[perm[i]]
                if directions & (1 << i):
                    x1, y1, x2, y2 = line
                else:
                    x2, y2, x1, y1 = line
                
                time += dist(x, y, x1, y1) / S
                time += line_length(x1, y1, x2, y2) / T
                x, y = x2, y2
            
            best_time = min(best_time, time)
    
    return best_time

# Read input
N, S, T = map(int, input().split())
lines = [tuple(map(int, input().split())) for _ in range(N)]

# Solve and print result
result = solve(N, S, T, lines)
print(f"{result:.20f}")