def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    coords = []
    idx = 1
    for i in range(N):
        x = int(input_data[idx])
        y = int(input_data[idx+1])
        coords.append((x, y))
        idx += 2

    results = []
    for i in range(N):
        xi, yi = coords[i]
        max_distance_sq = -1
        farthest_id = None
        for j in range(N):
            if i == j:
                continue
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            distance_sq = dx*dx + dy*dy
            if distance_sq > max_distance_sq:
                max_distance_sq = distance_sq
                farthest_id = j + 1  # IDs are 1-indexed
            elif distance_sq == max_distance_sq:
                if j + 1 < farthest_id:
                    farthest_id = j + 1
        results.append(farthest_id)

    output = sys.stdout
    for result in results:
        output.write(str(result) + "
")

if __name__ == '__main__':
    main()