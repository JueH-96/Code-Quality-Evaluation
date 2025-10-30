def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    points = []
    idx = 1
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx+1])
        points.append((x, y))
        idx += 2

    for i in range(N):
        max_dist_sq = -float('inf')
        best_ids = []
        for j in range(N):
            if i == j:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                best_ids = [j + 1]
            elif dist_sq == max_dist_sq:
                best_ids.append(j + 1)
        print(min(best_ids))

if __name__ == '__main__':
    main()