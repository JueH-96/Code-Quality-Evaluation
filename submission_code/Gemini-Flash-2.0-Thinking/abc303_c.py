def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    items = set()
    for _ in range(m):
        x, y = map(int, input().split())
        items.add((x, y))

    current_x, current_y = 0, 0
    current_health = h

    for move in s:
        current_health -= 1
        if current_health < 0:
            print("No")
            return

        if move == 'R':
            current_x += 1
        elif move == 'L':
            current_x -= 1
        elif move == 'U':
            current_y += 1
        elif move == 'D':
            current_y -= 1

        current_pos = (current_x, current_y)
        if current_pos in items and current_health < k:
            current_health = k
            items.discard(current_pos)

    print("Yes")

solve()