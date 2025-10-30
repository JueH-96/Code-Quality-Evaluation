# YOUR CODE HERE
import sys
import math

def find_farthest_points(n, points):
    results = []
    for i in range(n):
        max_distance = -1
        farthest_id = -1
        for j in range(n):
            if i != j:
                # Calculate the Euclidean distance
                distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_id = j + 1
                elif distance == max_distance:
                    farthest_id = min(farthest_id, j + 1)
        results.append(farthest_id)
    return results

def main():
    input = sys.stdin.read
    data = input().strip().split()
    
    n = int(data[0])
    points = []
    
    index = 1
    for _ in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    
    results = find_farthest_points(n, points)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()