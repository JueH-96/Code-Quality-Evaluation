def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = []
    for i in range(N):
        x = int(data[1 + 2*i])
        y = int(data[2 + 2*i])
        points.append((x, y))
    
    for i in range(N):
        max_dist = -1
        farthest_id = -1
        for j in range(N):
            if i == j:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > max_dist:
                max_dist = dist_sq
                farthest_id = j + 1  # IDs start from 1
            elif dist_sq == max_dist:
                farthest_id = min(farthest_id, j + 1)
        print(farthest_id)

if __name__ == "__main__":
    main()