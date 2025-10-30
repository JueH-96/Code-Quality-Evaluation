import sys

def main():
    N = int(sys.stdin.readline())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
    
    for i in range(N):
        max_dist_sq = -1
        farthest_id = -1
        xi, yi = points[i]
        for j in range(N):
            if i == j:
                continue
            xj, yj = points[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            if dist_sq > max_dist_sq or (dist_sq == max_dist_sq and j+1 < farthest_id):
                max_dist_sq = dist_sq
                farthest_id = j + 1
        print(farthest_id)

if __name__ == "__main__":
    main()