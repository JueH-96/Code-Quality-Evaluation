# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = []
    idx = 1
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        coords.append((x, y))
        idx += 2

    for i in range(N):
        x_i, y_i = coords[i]
        max_dist_sq = -1
        farthest_id = -1
        for j in range(N):
            if i == j:
                continue
            x_j, y_j = coords[j]
            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                farthest_id = j + 1  # IDs are 1-based
            elif dist_sq == max_dist_sq:
                farthest_id = min(farthest_id, j + 1)
        print(farthest_id)

# Don't forget to call main()!
main()