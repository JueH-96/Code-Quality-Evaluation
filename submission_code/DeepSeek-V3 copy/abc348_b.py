import math

def main():
    N = int(input())
    points = []
    for i in range(N):
        x, y = map(int, input().split())
        points.append((x, y, i+1))  # storing ID as i+1
    
    for i in range(N):
        max_dist = -1
        farthest_id = N + 1  # Initialize with a value larger than any possible ID
        for j in range(N):
            if i == j:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > max_dist:
                max_dist = dist_sq
                farthest_id = points[j][2]
            elif dist_sq == max_dist:
                if points[j][2] < farthest_id:
                    farthest_id = points[j][2]
        print(farthest_id)

if __name__ == "__main__":
    main()