import heapq

def solve():
    H, W = map(int, input().split())
    F = []
    for _ in range(H):
        F.append(list(map(int, input().split())))
    Q = int(input())

    def solve_query(A, B, Y, C, D, Z):
        A -= 1
        B -= 1
        C -= 1
        D -= 1

        dist = {}
        pq = []

        dist[(A, B, Y)] = 0
        heapq.heappush(pq, (0, A, B, Y))

        while pq:
            d, r, c, floor = heapq.heappop(pq)

            if d > dist.get((r, c, floor), float('inf')):
                continue

            # Move up or down
            if floor > 1:
                new_floor = floor - 1
                new_dist = d + 1
                if new_dist < dist.get((r, c, new_floor), float('inf')):
                    dist[(r, c, new_floor)] = new_dist
                    heapq.heappush(pq, (new_dist, r, c, new_floor))
            if floor < F[r][c]:
                new_floor = floor + 1
                new_dist = d + 1
                if new_dist < dist.get((r, c, new_floor), float('inf')):
                    dist[(r, c, new_floor)] = new_dist
                    heapq.heappush(pq, (new_dist, r, c, new_floor))

            # Move to adjacent blocks
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if r < H - 1:
                neighbors.append((r + 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            if c < W - 1:
                neighbors.append((r, c + 1))

            for nr, nc in neighbors:
                if F[nr][nc] >= floor:
                    new_dist = d
                    if new_dist < dist.get((nr, nc, floor), float('inf')):
                        dist[(nr, nc, floor)] = new_dist
                        heapq.heappush(pq, (new_dist, nr, nc, floor))

        return dist.get((C, D, Z), float('inf'))

    for _ in range(Q):
        A, B, Y, C, D, Z = map(int, input().split())
        print(solve_query(A, B, Y, C, D, Z))

solve()