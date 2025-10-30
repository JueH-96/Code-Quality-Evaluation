# YOUR CODE HERE
import sys
import math

def find_farthest_points(points):
    n = len(points)
    farthest_ids = [0] * n
    
    for i in range(n):
        max_distance = -1
        farthest_id = -1
        for j in range(n):
            if i != j:
                distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance > max_distance or (distance == max_distance and j < farthest_id):
                    max_distance = distance
                    farthest_id = j
        farthest_ids[i] = farthest_id + 1  # Convert to 1-based index
    
    return farthest_ids

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    points = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(n)]
    
    farthest_ids = find_farthest_points(points)
    
    for id in farthest_ids:
        print(id)

if __name__ == "__main__":
    main()