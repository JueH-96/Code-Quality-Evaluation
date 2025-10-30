def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = []
    idx = 1
    for i in range(N):
        x = int(input_data[idx]); y = int(input_data[idx+1])
        coords.append((x, y))
        idx += 2

    def dist_squared(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    for i in range(N):
        max_dist_sq = -1
        ans_id = -1
        for j in range(N):
            if i == j:
                continue
            d_sq = dist_squared(coords[i], coords[j])
            if d_sq > max_dist_sq:
                max_dist_sq = d_sq
                ans_id = j + 1  # +1 because IDs are 1-based
            elif d_sq == max_dist_sq:
                ans_id = min(ans_id, j + 1)
        print(ans_id)

# Do not forget to call main()
if __name__ == "__main__":
    main()