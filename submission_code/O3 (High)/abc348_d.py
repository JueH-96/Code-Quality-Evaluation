import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().splitlines()
    it = iter(input_data)

    H, W = map(int, next(it).split())
    grid = []
    sr = sc = tr = tc = -1
    for i in range(H):
        row = list(next(it))
        for j, ch in enumerate(row):
            if ch == 'S':
                sr, sc = i, j
            elif ch == 'T':
                tr, tc = i, j
        grid.append(row)

    N = int(next(it))
    medicine = {}                         # (r, c) -> E
    for _ in range(N):
        r, c, e = map(int, next(it).split())
        medicine[(r - 1, c - 1)] = e      # zero–based indices

    # initial energy (use medicine at the start cell if any)
    start_energy = medicine.get((sr, sc), 0)

    # If we cannot move at all and start is not the goal, impossible
    if start_energy == 0 and (sr, sc) != (tr, tc):
        print("No")
        return

    # best remaining energy seen so far for each cell
    best = [[-1] * W for _ in range(H)]
    best[sr][sc] = start_energy

    # max-heap (negative energy for heapq)
    pq = [(-start_energy, sr, sc)]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        neg_e, r, c = heapq.heappop(pq)
        e = -neg_e

        # outdated entry ?
        if e < best[r][c]:
            continue

        # reached goal
        if (r, c) == (tr, tc):
            print("Yes")
            return

        if e == 0:                # cannot move further
            continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr][nc] == '#':
                continue

            ne = e - 1            # spend 1 energy to move
            # use medicine on the destination cell if it increases energy
            med = medicine.get((nr, nc))
            if med is not None and med > ne:
                ne = med

            if ne > best[nr][nc]:
                best[nr][nc] = ne
                heapq.heappush(pq, (-ne, nr, nc))

    print("No")


if __name__ == "__main__":
    main()