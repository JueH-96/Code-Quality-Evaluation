n, q = map(int, input().split())
dragon = [(i, 0) for i in range(1, n + 1)]

for _ in range(q):
    query = input().split()
    type = int(query[0])

    if type == 1:
        direction = query[1]
        head_x, head_y = dragon[0]

        if direction == 'R':
            new_head = (head_x + 1, head_y)
        elif direction == 'L':
            new_head = (head_x - 1, head_y)
        elif direction == 'U':
            new_head = (head_x, head_y + 1)
        elif direction == 'D':
            new_head = (head_x, head_y - 1)

        dragon.insert(0, new_head)
        dragon.pop()

    else:
        p = int(query[1])
        print(dragon[p - 1][0], dragon[p - 1][1])