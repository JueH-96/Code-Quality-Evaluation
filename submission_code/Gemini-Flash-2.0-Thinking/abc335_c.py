def solve():
    n, q = map(int, input().split())
    positions = [(i + 1, 0) for i in range(n)]

    for _ in range(q):
        query = input().split()
        if query[0] == '1':
            direction = query[1]
            old_head_pos = positions[0]
            new_head_x, new_head_y = positions[0]

            if direction == 'R':
                new_head_x += 1
            elif direction == 'L':
                new_head_x -= 1
            elif direction == 'U':
                new_head_y += 1
            elif direction == 'D':
                new_head_y -= 1

            prev_pos = old_head_pos
            positions[0] = (new_head_x, new_head_y)

            for i in range(1, n):
                current_pos = positions[i]
                positions[i] = prev_pos
                prev_pos = current_pos

        elif query[0] == '2':
            p = int(query[1])
            print(positions[p-1][0], positions[p-1][1])

solve()