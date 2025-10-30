import sys
import math
from itertools import combinations

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def min_distance_with_penalty(N, checkpoints):
    min_distance = float('inf')
    
    # Check all possible subsets of checkpoints to skip
    for r in range(N):
        for subset in combinations(range(1, N-1), r):
            total_distance = 0
            prev_checkpoint = 0
            for checkpoint in subset:
                total_distance += euclidean_distance(checkpoints[prev_checkpoint][0], checkpoints[prev_checkpoint][1], checkpoints[checkpoint][0], checkpoints[checkpoint][1])
                prev_checkpoint = checkpoint
            total_distance += euclidean_distance(checkpoints[prev_checkpoint][0], checkpoints[prev_checkpoint][1], checkpoints[N-1][0], checkpoints[N-1][1])
            penalty = 2 ** (r - 1) if r > 0 else 0
            total_distance += penalty
            min_distance = min(min_distance, total_distance)
    
    return min_distance

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
checkpoints = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

# Calculate and print the result
result = min_distance_with_penalty(N, checkpoints)
print(result)