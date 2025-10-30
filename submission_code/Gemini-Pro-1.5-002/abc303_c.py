# YOUR CODE HERE
def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    items = []
    for _ in range(m):
        items.append(tuple(map(int, input().split())))

    current_health = h
    current_pos = (0, 0)
    item_locations = set(items)

    for move in s:
        if move == 'R':
            current_pos = (current_pos[0] + 1, current_pos[1])
        elif move == 'L':
            current_pos = (current_pos[0] - 1, current_pos[1])
        elif move == 'U':
            current_pos = (current_pos[0], current_pos[1] + 1)
        elif move == 'D':
            current_pos = (current_pos[0], current_pos[1] - 1)
        
        current_health -= 1

        if current_health < 0:
            print("No")
            return

        if current_pos in item_locations and current_health < k:
            current_health = k

    print("Yes")


solve()