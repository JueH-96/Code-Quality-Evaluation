import heapq

def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())

    def get_tile(x, y):
        if (x + y) % 2 == 0:
            return (x // 2, y)
        else:
            return ((x + 1) // 2, y)

    start_tile = get_tile(sx, sy)
    end_tile = get_tile(tx, ty)

    if start_tile == end_tile:
        print(0)
        return

    q = [(0, start_tile)]
    visited = {start_tile: 0}

    while q:
        cost, current_tile = heapq.heappop(q)

        if current_tile == end_tile:
            print(cost)
            return

        x, y = current_tile
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]

        for nx, ny in neighbors:
            neighbor_tile = (nx, ny)
            new_cost = cost + 1
            
            if neighbor_tile not in visited or new_cost < visited[neighbor_tile]:
                visited[neighbor_tile] = new_cost
                heapq.heappush(q, (new_cost, neighbor_tile))

solve()