import math
from itertools import combinations

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

N = int(input())
X = []
Y = []
for _ in range(N):
    x, y = map(int, input().split())
    X.append(x)
    Y.append(y)

min_cost = float('inf')

# Try all possible combinations of checkpoints to skip
for skip_count in range(N-1):
    # Get all possible combinations of indices to skip
    # Skip indices 1 to N-2 (0-based indexing)
    middle_points = list(range(1, N-1))
    for skip_points in combinations(middle_points, skip_count):
        # Create path by keeping required points and points not in skip_points
        path = [0] # Start with first checkpoint
        for i in range(1, N-1):
            if i not in skip_points:
                path.append(i)
        path.append(N-1) # End with last checkpoint
        
        # Calculate total distance
        total_dist = 0
        for i in range(len(path)-1):
            curr = path[i]
            next_point = path[i+1]
            total_dist += dist(X[curr], Y[curr], X[next_point], Y[next_point])
        
        # Add penalty if points were skipped
        penalty = 0
        if skip_count > 0:
            penalty = 2**(skip_count-1)
            
        min_cost = min(min_cost, total_dist + penalty)

print(min_cost)