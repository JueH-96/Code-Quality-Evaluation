import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W, Y = map(int, input().split())
    total = H * W

    # pos[h] will hold the flat indices of cells whose elevation == h (for h <= Y)
    pos = [[] for _ in range(Y+1)]

    # Read the grid, flatten into Aflat, and fill pos
    Aflat = [0] * total
    for i in range(H):
        row = list(map(int, input().split()))
        base = i * W
        for j, v in enumerate(row):
            idx = base + j
            Aflat[idx] = v
            if v <= Y:
                pos[v].append(idx)

    # water[idx] == 1 means that cell has sunk (is sea). Use bytearray for memory & speed.
    water = bytearray(total)
    water_count = 0

    ans = [0] * (Y+1)
    dq = deque()

    # Directions in flat index arithmetic:
    # up: idx - W, down: idx + W, left: idx - 1, right: idx + 1

    for level in range(1, Y+1):
        # Mark newly eligible cells at this elevation
        for idx in pos[level]:
            if water[idx]:
                continue
            # Compute i,j from idx
            i = idx // W
            j = idx - i * W

            # Check if it should immediately sink:
            # 1) if it's on the border
            # 2) or any neighbor is already water
            is_sea = (i == 0 or i == H-1 or j == 0 or j == W-1)
            if not is_sea:
                # check four neighbors
                if water[idx - W] or water[idx + W] or (j > 0 and water[idx - 1]) or (j < W-1 and water[idx + 1]):
                    is_sea = True

            if is_sea:
                water[idx] = 1
                water_count += 1
                dq.append(idx)

        # BFS to flood all reachable eligible cells
        while dq:
            idx = dq.popleft()
            i = idx // W
            j = idx - i * W

            # up
            if i > 0:
                nid = idx - W
                if not water[nid] and Aflat[nid] <= level:
                    water[nid] = 1
                    water_count += 1
                    dq.append(nid)
            # down
            if i < H-1:
                nid = idx + W
                if not water[nid] and Aflat[nid] <= level:
                    water[nid] = 1
                    water_count += 1
                    dq.append(nid)
            # left
            if j > 0:
                nid = idx - 1
                if not water[nid] and Aflat[nid] <= level:
                    water[nid] = 1
                    water_count += 1
                    dq.append(nid)
            # right
            if j < W-1:
                nid = idx + 1
                if not water[nid] and Aflat[nid] <= level:
                    water[nid] = 1
                    water_count += 1
                    dq.append(nid)

        # Area that remains above sea level = total cells - flooded cells
        ans[level] = total - water_count

    # Output answers for years 1..Y
    out = sys.stdout.write
    for level in range(1, Y+1):
        out(str(ans[level]) + "
")


if __name__ == "__main__":
    main()