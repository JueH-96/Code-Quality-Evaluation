import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    points = [(int(next(it)), int(next(it))) for _ in range(N)]

    for i in range(N):
        xi, yi = points[i]
        max_dist2 = -1
        best_id = 0
        for j in range(N):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist2 = dx * dx + dy * dy
            if dist2 > max_dist2 or (dist2 == max_dist2 and j + 1 < best_id):
                max_dist2 = dist2
                best_id = j + 1  # IDs are 1-based
        print(best_id)

if __name__ == "__main__":
    main()