def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    points = []
    index = 1
    for _ in range(n):
        x = int(input_data[index])
        y = int(input_data[index + 1])
        points.append((x, y))
        index += 2

    # For each point, find the farthest point.
    results = []
    for i in range(n):
        xi, yi = points[i]
        max_dist_sq = -1
        farthest_point_id = -1  # 1-indexed id
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy  # squared distance
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                farthest_point_id = j + 1
            elif dist_sq == max_dist_sq:
                # Tie: choose the smallest ID
                farthest_point_id = min(farthest_point_id, j + 1)
        results.append(farthest_point_id)

    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()