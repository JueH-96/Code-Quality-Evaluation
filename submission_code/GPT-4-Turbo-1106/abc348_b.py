import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_farthest_points(N, points):
    for i in range(N):
        max_dist = -1
        farthest_point_id = -1
        for j in range(N):
            if i != j:
                dist = distance(points[i], points[j])
                if dist > max_dist or (dist == max_dist and j < farthest_point_id):
                    max_dist = dist
                    farthest_point_id = j
        print(farthest_point_id + 1)

def main():
    N = int(input().strip())
    points = [tuple(map(int, input().strip().split())) for _ in range(N)]
    find_farthest_points(N, points)

if __name__ == "__main__":
    main()