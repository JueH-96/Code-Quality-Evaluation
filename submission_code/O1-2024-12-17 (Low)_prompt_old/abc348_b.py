def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = []
    idx = 1
    for i in range(N):
        x = int(input_data[idx]); y = int(input_data[idx+1])
        coords.append((x, y))
        idx += 2

    for i in range(N):
        x1, y1 = coords[i]
        max_dist = -1
        farthest_id = -1
        for j in range(N):
            if i == j:
                continue
            x2, y2 = coords[j]
            # Euclidean distance squared (avoid sqrt for performance and correctness)
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq > max_dist:
                max_dist = dist_sq
                farthest_id = j + 1
            elif dist_sq == max_dist:
                # If the same distance, choose smaller ID
                if j + 1 < farthest_id:
                    farthest_id = j + 1
        print(farthest_id)

# Call solve() to execute
solve()