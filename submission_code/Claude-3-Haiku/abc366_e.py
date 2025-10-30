from collections import defaultdict

def count_points(N, D, points):
    # Create a dictionary to store the sum of Manhattan distances for each point
    distance_sums = defaultdict(int)
    
    for x, y in points:
        for i in range(-D, D+1):
            for j in range(-D, D+1):
                if abs(i) + abs(j) <= D:
                    distance_sums[(x+i, y+j)] += 1
    
    return sum(distance_sums.values())

# Read input
N, D = map(int, input().split())
points = [tuple(map(int, input().split())) for _ in range(N)]

# Solve the problem
result = count_points(N, D, points)
print(result)