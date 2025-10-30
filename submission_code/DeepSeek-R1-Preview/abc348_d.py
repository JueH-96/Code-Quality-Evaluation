import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    grid = []
    start = None
    goal = None
    for i in range(H):
        line = input[ptr]
        ptr += 1
        grid.append(list(line))
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    N = int(input[ptr])
    ptr += 1
    medicine = {}
    for _ in range(N):
        R = int(input[ptr]) - 1
        ptr += 1
        C = int(input[ptr]) - 1
        ptr += 1
        E = int(input[ptr])
        ptr += 1
        medicine[(R, C)] = E

    max_energy = [[-1 for _ in range(W)] for _ in range(H)]
    si, sj = start
    max_energy[si][sj] = 0
    heap = []
    heapq.heappush(heap, (0, si, sj))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    found = False

    while heap:
        e_neg, i, j = heapq.heappop(heap)
        e = -e_neg

        if (i, j) == goal:
            found = True
            break

        if e < max_energy[i][j]:
            continue

        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    continue
                new_e = e - 1
                if new_e >= 0 and new_e > max_energy[ni][nj]:
                    max_energy[ni][nj] = new_e
                    heapq.heappush(heap, (-new_e, ni, nj))

        if (i, j) in medicine:
            new_e = medicine[(i, j)]
            if new_e > max_energy[i][j]:
                max_energy[i][j] = new_e
                heapq.heappush(heap, (-new_e, i, j))

    print("Yes" if found else "No")

if __name__ == '__main__':
    main()