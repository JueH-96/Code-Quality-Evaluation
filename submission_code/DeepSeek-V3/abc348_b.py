# YOUR CODE HERE
import math

def main():
    N = int(input())
    points = []
    for i in range(N):
        x, y = map(int, input().split())
        points.append((x, y, i+1))  # storing ID as i+1
    
    for i in range(N):
        max_dist = -1
        farthest_id = -1
        x1, y1, _ = points[i]
        for j in range(N):
            if i == j:
                continue
            x2, y2, id2 = points[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq > max_dist:
                max_dist = dist_sq
                farthest_id = id2
            elif dist_sq == max_dist:
                if id2 < farthest_id:
                    farthest_id = id2
        print(farthest_id)

if __name__ == "__main__":
    main()