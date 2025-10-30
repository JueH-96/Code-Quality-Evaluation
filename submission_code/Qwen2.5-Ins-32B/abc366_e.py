import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    n, D = int(data[0]), int(data[1])
    points = [(int(data[i]), int(data[i+1])) for i in range(2, 2*n, 2)]
    
    # Calculate the sum of distances for each point
    dist_sum_x = defaultdict(int)
    dist_sum_y = defaultdict(int)
    for x, y in points:
        dist_sum_x[x] += 1
        dist_sum_y[y] += 1
    
    # Calculate the prefix sums for x and y coordinates
    prefix_sum_x = defaultdict(int)
    prefix_sum_y = defaultdict(int)
    for x in sorted(dist_sum_x.keys()):
        prefix_sum_x[x] = prefix_sum_x[prev_x] + dist_sum_x[x] if prev_x else dist_sum_x[x]
        prev_x = x
    
    for y in sorted(dist_sum_y.keys()):
        prefix_sum_y[y] = prefix_sum_y[prev_y] + dist_sum_y[y] if prev_y else dist_sum_y[y]
        prev_y = y
    
    # Calculate the number of valid (x, y) pairs
    count = 0
    for x in range(-2*10**6, 2*10**6 + 1):
        for y in range(-2*10**6, 2*10**6 + 1):
            total_dist = 0
            for px, py in points:
                total_dist += abs(x - px) + abs(y - py)
            if total_dist <= D:
                count += 1
    
    print(count)

solve()