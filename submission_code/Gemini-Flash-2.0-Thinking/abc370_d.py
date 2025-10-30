def solve():
    h, w, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(tuple(map(int, input().split())))

    walls = set()
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            walls.add((r, c))

    def destroy_first_walls(r, c, walls, h, w):
        destroyed_walls = set()

        # Up
        for i in range(r - 1, 0, -1):
            if (i, c) in walls:
                destroyed_walls.add((i, c))
                break

        # Down
        for i in range(r + 1, h + 1):
            if (i, c) in walls:
                destroyed_walls.add((i, c))
                break

        # Left
        for j in range(c - 1, 0, -1):
            if (r, j) in walls:
                destroyed_walls.add((r, j))
                break

        # Right
        for j in range(c + 1, w + 1):
            if (r, j) in walls:
                destroyed_walls.add((r, j))
                break

        walls.difference_update(destroyed_walls)

    for r_q, c_q in queries:
        if (r_q, c_q) in walls:
            walls.remove((r_q, c_q))
        else:
            destroy_first_walls(r_q, c_q, walls, h, w)

    print(len(walls))

# YOUR CODE HERE
solve()