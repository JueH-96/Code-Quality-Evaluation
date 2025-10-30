def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, D = map(int, data[:3])
    grid = data[3:]

    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))

    def manhattan(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    max_humidified = 0
    n = len(floors)
    for i in range(n):
        for j in range(i+1, n):
            (r1, c1) = floors[i]
            (r2, c2) = floors[j]
            count = 0
            for (r, c) in floors:
                if manhattan((r, c), (r1, c1)) <= D or manhattan((r, c), (r2, c2)) <= D:
                    count += 1
            max_humidified = max(max_humidified, count)

    print(max_humidified)

# Do not remove the below line
if __name__ == "__main__":
    main()