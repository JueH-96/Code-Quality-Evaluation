# YOUR CODE HERE
n, m, h, k = map(int, input().split())
s = input()
items = [tuple(map(int, input().split())) for _ in range(m)]
health = h
position = (0, 0)
for move in s:
    if move == 'R':
        position = (position[0] + 1, position[1])
    elif move == 'L':
        position = (position[0] - 1, position[1])
    elif move == 'U':
        position = (position[0], position[1] + 1)
    elif move == 'D':
        position = (position[0], position[1] - 1)
    health -= 1
    if health <= 0:
        print('No')
        break
    for item in items:
        if position == item and health < k:
            health = k
else:
    print('Yes')